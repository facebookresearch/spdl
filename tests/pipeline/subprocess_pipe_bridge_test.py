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
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest import mock

from spdl.pipeline import AsyncQueue
from spdl.pipeline._components import _subprocess_pipe
from spdl.pipeline._components._common import _EOF, _EPOCH_END, StageInfo
from spdl.pipeline._subprocess_pipeline_pool import _SerializationFailureState


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


def _serialization_failure_state(
    *,
    failed: bool = False,
    relay_failed: bool = False,
    non_payload_failed: bool = False,
) -> _SerializationFailureState:
    failed_event = threading.Event()
    relay_failed_event = threading.Event()
    non_payload_failed_event = threading.Event()
    if failed:
        failed_event.set()
    if relay_failed:
        relay_failed_event.set()
    if non_payload_failed:
        non_payload_failed_event.set()
    return _SerializationFailureState(
        failed_event, relay_failed_event, non_payload_failed_event
    )


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


class _BusyResultQueue:
    """Return results without an empty poll, then expose an empty queue."""

    def __init__(self) -> None:
        self.calls = 0

    def get(self, *, timeout: float) -> tuple[int, list[int]]:
        del timeout
        self.calls += 1
        if self.calls <= 2:
            return (_subprocess_pipe._RESULT, [self.calls])
        raise _queue.Empty

    def get_nowait(self) -> Any:
        raise _queue.Empty


class _ScriptedResultQueue:
    """Return scripted blocking reads while exposing no buffered messages."""

    def __init__(self, reads: list[tuple[int, Any] | None]) -> None:
        self._reads = reads
        self.calls = 0

    def get(self, *, timeout: float) -> tuple[int, Any]:
        del timeout
        self.calls += 1
        if not self._reads:
            raise AssertionError("collector read past the scripted messages")
        result = self._reads.pop(0)
        if result is None:
            raise _queue.Empty
        return result

    def get_nowait(self) -> Any:
        raise _queue.Empty


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
            self.assertEqual(worker[epoch_at], (_subprocess_pipe._EPOCH, 0))
            self.assertTrue(all(i < 100 for i in _items_of(worker[:epoch_at])))
            self.assertTrue(all(i >= 100 for i in _items_of(worker[epoch_at:])))

    def test_epoch_markers_carry_monotonic_generations(self) -> None:
        """Every broadcast identifies its epoch independently of barrier timing."""
        msgs = self._run_feed_continuous(
            [_EPOCH_END, _EPOCH_END], num_workers=1, buffer_size=1
        )

        self.assertEqual(
            [payload for kind, payload in msgs[0] if kind == _subprocess_pipe._EPOCH],
            [0, 1],
        )


class CollectUnpackTest(unittest.TestCase):
    """The collector unpacks a transfer back into individual downstream items."""

    def _assert_malformed_message_rejected(
        self, message: Any, *, continuous: bool
    ) -> str:
        async def _scenario() -> None:
            out_q: _queue.Queue[Any] = _queue.Queue()
            out_q.put(message)
            with ThreadPoolExecutor(max_workers=1) as ex:
                if continuous:
                    await _subprocess_pipe._collect_continuous(
                        out_q,
                        1,
                        _unbounded_queue("output"),
                        ex,
                        asyncio.Event(),
                        asyncio.Event(),
                    )
                else:
                    abort = asyncio.Event()
                    try:
                        await _subprocess_pipe._collect(
                            out_q,
                            1,
                            _unbounded_queue("output"),
                            ex,
                            abort,
                            asyncio.Event(),
                        )
                    finally:
                        self.assertTrue(abort.is_set())

        with self.assertRaises(RuntimeError) as raised:
            asyncio.run(_scenario())
        detail = str(raised.exception)
        self.assertIn("malformed message", detail)
        return detail

    def test_collectors_reject_unknown_message_kinds(self) -> None:
        """Unknown protocol tags cannot be silently dropped."""
        for continuous in (False, True):
            with self.subTest(continuous=continuous):
                detail = self._assert_malformed_message_rejected(
                    (99, None), continuous=continuous
                )
                self.assertIn("unexpected message kind 99", detail)

    def test_collectors_reject_non_list_result_payloads(self) -> None:
        """Malformed result payloads raise a protocol error before iteration."""
        for continuous in (False, True):
            with self.subTest(continuous=continuous):
                detail = self._assert_malformed_message_rejected(
                    (_subprocess_pipe._RESULT, None),
                    continuous=continuous,
                )
                self.assertIn("RESULT payload must be a list, got NoneType", detail)

    def test_collectors_describe_malformed_protocol_structure(self) -> None:
        """Protocol failures identify shape and payload class without rendering payloads."""
        cases = (
            (object(), "expected a 2-tuple, got object"),
            ((1,), "expected 2 fields, got 1"),
            (("result", []), "message kind must be an integer, got str"),
            (
                (10**10_000, None),
                "unexpected integer message kind outside 64-bit range",
            ),
            (
                (_subprocess_pipe._ERROR, "not an exception"),
                "ERROR payload must be an exception, got str",
            ),
            (
                (_subprocess_pipe._DONE, "not empty"),
                "control-message payload must be None, got str",
            ),
        )
        for continuous in (False, True):
            for message, expected in cases:
                with self.subTest(continuous=continuous, expected=expected):
                    detail = self._assert_malformed_message_rejected(
                        message, continuous=continuous
                    )
                    self.assertIn(expected, detail)

    def test_malformed_protocol_type_diagnostics_are_bounded_and_safe(self) -> None:
        """Hostile type metadata cannot escape or create an unbounded protocol error."""

        class _LongNameMeta(type):
            def __getattribute__(cls, name: str) -> Any:
                if name == "__name__":
                    return "x" * 10_000
                return super().__getattribute__(name)

        class _BrokenNameMeta(type):
            def __getattribute__(cls, name: str) -> Any:
                if name == "__name__":
                    raise RuntimeError("type name failed")
                return super().__getattribute__(name)

        class _HostileName(str):
            def __len__(self) -> int:
                raise RuntimeError("type name length failed")

        class _HostileStringNameMeta(type):
            def __getattribute__(cls, name: str) -> Any:
                if name == "__name__":
                    return _HostileName("hostile name")
                return super().__getattribute__(name)

        class _HostileTuple(tuple[Any, ...]):
            def __len__(self) -> int:
                raise RuntimeError("message length failed")

        class _HostileClassPayload:
            def __getattribute__(self, name: str) -> Any:
                if name == "__class__":
                    raise RuntimeError("class lookup failed")
                return super().__getattribute__(name)

        class _HostileList(list[Any]):
            def __iter__(self) -> Iterator[Any]:
                raise RuntimeError("payload iteration failed")

        class _LongNamePayload(metaclass=_LongNameMeta):
            pass

        class _BrokenNamePayload(metaclass=_BrokenNameMeta):
            def __repr__(self) -> str:
                raise RuntimeError("payload rendering failed")

        class _HostileStringNamePayload(metaclass=_HostileStringNameMeta):
            pass

        cases = (
            (_LongNamePayload(), "... <truncated>"),
            (_BrokenNamePayload(), "<type unavailable>"),
            (_HostileStringNamePayload(), "<type unavailable>"),
            (_HostileClassPayload(), "_HostileClassPayload"),
            (_HostileList(), "_HostileList"),
        )
        for payload, expected in cases:
            with self.subTest(expected=expected):
                detail = self._assert_malformed_message_rejected(
                    (_subprocess_pipe._RESULT, payload), continuous=False
                )
                self.assertIn(expected, detail)
                self.assertLess(len(detail), 512)

        hostile_message = _HostileTuple((_subprocess_pipe._RESULT, []))
        detail = self._assert_malformed_message_rejected(
            hostile_message, continuous=False
        )
        self.assertIn("expected a 2-tuple, got _HostileTuple", detail)

        detail = self._assert_malformed_message_rejected(
            (_subprocess_pipe._ERROR, _HostileClassPayload()), continuous=False
        )
        self.assertIn("ERROR payload must be an exception", detail)

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
            out_q.put((_subprocess_pipe._EPOCH_DONE, (0, 0)))
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

    def test_first_error_wins_while_remaining_messages_are_drained(self) -> None:
        """Post-error results are dropped and a later error cannot replace the first."""

        async def _scenario() -> tuple[list[Any], bool, bool]:
            out_q: _queue.Queue[Any] = _queue.Queue()
            out_q.put((_subprocess_pipe._RESULT, [1, 2]))
            out_q.put((_subprocess_pipe._ERROR, RuntimeError("first error")))
            out_q.put((_subprocess_pipe._RESULT, [3, 4]))
            out_q.put((_subprocess_pipe._ERROR, RuntimeError("second error")))
            out_q.put((_subprocess_pipe._DONE, None))
            out_q.put((_subprocess_pipe._DONE, None))
            output_queue = _unbounded_queue("output")
            abort = asyncio.Event()
            with ThreadPoolExecutor(max_workers=2) as ex:
                with self.assertRaisesRegex(RuntimeError, "first error"):
                    await asyncio.wait_for(
                        _subprocess_pipe._collect(
                            out_q, 2, output_queue, ex, abort, asyncio.Event()
                        ),
                        timeout=10.0,
                    )
            return (
                [output_queue.get_nowait() for _ in range(output_queue.qsize())],
                abort.is_set(),
                out_q.empty(),
            )

        self.assertEqual(asyncio.run(_scenario()), ([1, 2], True, True))


class SerializationMarkerOrderingTest(unittest.TestCase):
    """A queue marker cannot overtake a feeder's fallback serialization error."""

    def test_collect_waits_for_feeder_error_after_done(self) -> None:
        """A flagged serialization gap keeps ``_DONE`` from hiding its late error."""

        async def _scenario() -> None:
            out_q: _queue.Queue[Any] = _queue.Queue()
            out_q.put((_subprocess_pipe._DONE, None))
            out_q.put((_subprocess_pipe._ERROR, RuntimeError("late feeder error")))
            serialization_failed = _serialization_failure_state(failed=True)
            abort = asyncio.Event()
            with ThreadPoolExecutor(max_workers=2) as ex:
                with self.assertRaisesRegex(RuntimeError, "late feeder error"):
                    await asyncio.wait_for(
                        _subprocess_pipe._collect(
                            out_q,
                            1,
                            _unbounded_queue("output"),
                            ex,
                            abort,
                            asyncio.Event(),
                            (("input", serialization_failed),),
                        ),
                        timeout=10.0,
                    )
            self.assertTrue(abort.is_set())

        asyncio.run(_scenario())

    def test_continuous_collect_holds_boundary_for_late_feeder_error(self) -> None:
        """A flagged gap prevents a successful epoch boundary from overtaking its error."""

        async def _scenario() -> None:
            out_q: _queue.Queue[Any] = _queue.Queue()
            out_q.put((_subprocess_pipe._EPOCH_DONE, (0, 0)))
            out_q.put((_subprocess_pipe._ERROR, RuntimeError("late feeder error")))
            serialization_failed = _serialization_failure_state(failed=True)
            output_queue = _unbounded_queue("output")
            epoch_barrier = asyncio.Event()
            with ThreadPoolExecutor(max_workers=2) as ex:
                with self.assertRaisesRegex(RuntimeError, "late feeder error"):
                    await asyncio.wait_for(
                        _subprocess_pipe._collect_continuous(
                            out_q,
                            1,
                            output_queue,
                            ex,
                            epoch_barrier,
                            asyncio.Event(),
                            (("output", serialization_failed),),
                        ),
                        timeout=10.0,
                    )
            self.assertFalse(epoch_barrier.is_set())
            self.assertEqual(output_queue.qsize(), 0)

        asyncio.run(_scenario())

    def test_malformed_message_waits_for_pending_feeder_error(self) -> None:
        """Malformed queue noise cannot replace non-continuous feeder detail."""

        async def _scenario() -> None:
            out_q: _queue.Queue[Any] = _queue.Queue()
            out_q.put(("malformed",))
            out_q.put((_subprocess_pipe._ERROR, RuntimeError("late feeder error")))
            out_q.put((_subprocess_pipe._DONE, None))
            serialization_failed = _serialization_failure_state(failed=True)
            with ThreadPoolExecutor(max_workers=2) as ex:
                with self.assertRaisesRegex(RuntimeError, "late feeder error"):
                    await _subprocess_pipe._collect(
                        out_q,
                        1,
                        _unbounded_queue("output"),
                        ex,
                        asyncio.Event(),
                        asyncio.Event(),
                        (("input", serialization_failed),),
                    )

        asyncio.run(_scenario())

    def test_continuous_malformed_message_waits_for_pending_feeder_error(
        self,
    ) -> None:
        """Malformed queue noise cannot replace continuous feeder detail."""

        async def _scenario() -> None:
            out_q: _queue.Queue[Any] = _queue.Queue()
            out_q.put(("malformed",))
            out_q.put((_subprocess_pipe._ERROR, RuntimeError("late feeder error")))
            serialization_failed = _serialization_failure_state(failed=True)
            with ThreadPoolExecutor(max_workers=2) as ex:
                with self.assertRaisesRegex(RuntimeError, "late feeder error"):
                    await _subprocess_pipe._collect_continuous(
                        out_q,
                        1,
                        _unbounded_queue("output"),
                        ex,
                        asyncio.Event(),
                        asyncio.Event(),
                        (("output", serialization_failed),),
                    )

        asyncio.run(_scenario())

    def test_collect_prefers_buffered_error_after_relay_deadline(self) -> None:
        """An expired deadline cannot mask a detailed error behind ``_DONE``."""

        async def _scenario() -> None:
            out_q: _queue.Queue[Any] = _queue.Queue()
            out_q.put((_subprocess_pipe._RESULT, [1]))
            out_q.put((_subprocess_pipe._DONE, None))
            out_q.put((_subprocess_pipe._ERROR, RuntimeError("buffered detail")))
            serialization_failed = _serialization_failure_state(failed=True)
            with (
                ThreadPoolExecutor(max_workers=2) as ex,
                mock.patch.object(
                    _subprocess_pipe, "_SERIALIZATION_ERROR_RELAY_TIMEOUT", 0.0
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "buffered detail"):
                    await _subprocess_pipe._collect(
                        out_q,
                        1,
                        _unbounded_queue("output"),
                        ex,
                        asyncio.Event(),
                        asyncio.Event(),
                        (("output", serialization_failed),),
                    )

        asyncio.run(_scenario())

    def test_continuous_collect_prefers_buffered_error_after_deadline(self) -> None:
        """An expired deadline cannot mask detail behind ``_EPOCH_DONE``."""

        async def _scenario() -> None:
            out_q: _queue.Queue[Any] = _queue.Queue()
            out_q.put((_subprocess_pipe._RESULT, [1]))
            out_q.put((_subprocess_pipe._EPOCH_DONE, (0, 0)))
            out_q.put((_subprocess_pipe._ERROR, RuntimeError("buffered detail")))
            serialization_failed = _serialization_failure_state(failed=True)
            epoch_barrier = asyncio.Event()
            with (
                ThreadPoolExecutor(max_workers=2) as ex,
                mock.patch.object(
                    _subprocess_pipe, "_SERIALIZATION_ERROR_RELAY_TIMEOUT", 0.0
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "buffered detail"):
                    await _subprocess_pipe._collect_continuous(
                        out_q,
                        1,
                        _unbounded_queue("output"),
                        ex,
                        epoch_barrier,
                        asyncio.Event(),
                        (("output", serialization_failed),),
                    )
            self.assertFalse(epoch_barrier.is_set())

        asyncio.run(_scenario())

    def test_buffered_error_scan_skips_malformed_messages(self) -> None:
        """Malformed queue entries cannot hide a later detailed error."""
        out_q: _queue.Queue[Any] = _queue.Queue()
        out_q.put(object())
        out_q.put((_subprocess_pipe._ERROR,))
        detail = RuntimeError("buffered detail")
        out_q.put((_subprocess_pipe._ERROR, detail))

        self.assertIs(
            _subprocess_pipe._pop_buffered_serialization_error(out_q, 1.0), detail
        )

    def test_buffered_error_scan_ignores_malformed_only_messages(self) -> None:
        """A scan with no valid exception leaves the generic fallback intact."""
        out_q: _queue.Queue[Any] = _queue.Queue()
        out_q.put(object())
        out_q.put((_subprocess_pipe._ERROR,))
        out_q.put((_subprocess_pipe._ERROR, "not an exception"))

        self.assertIsNone(
            _subprocess_pipe._pop_buffered_serialization_error(out_q, 1.0)
        )

    def test_buffered_error_scan_skips_hostile_tuple_subclasses(self) -> None:
        """Hostile queue noise cannot escape or mask a later detailed error."""

        class _HostileTuple(tuple[Any, ...]):
            def __len__(self) -> int:
                raise RuntimeError("message length failed")

        out_q: _queue.Queue[Any] = _queue.Queue()
        out_q.put(_HostileTuple((_subprocess_pipe._ERROR, RuntimeError("hostile"))))
        detail = RuntimeError("buffered detail")
        out_q.put((_subprocess_pipe._ERROR, detail))

        self.assertIs(
            _subprocess_pipe._pop_buffered_serialization_error(out_q, 1.0), detail
        )

    def test_buffered_error_scan_is_not_limited_by_queue_capacity(self) -> None:
        """Concurrent producer messages cannot hide detail beyond one queue depth."""
        out_q: _queue.Queue[Any] = _queue.Queue()
        for value in range(_subprocess_pipe._fused_queue_capacity(1) + 3):
            out_q.put((_subprocess_pipe._RESULT, [value]))
        detail = RuntimeError("buffered detail")
        out_q.put((_subprocess_pipe._ERROR, detail))

        self.assertIs(
            _subprocess_pipe._pop_buffered_serialization_error(out_q, 1.0), detail
        )

    def test_buffered_error_scan_is_time_bounded(self) -> None:
        """A continuously replenished queue cannot make fallback recovery hang."""
        out_q = mock.Mock()
        out_q.get.return_value = (_subprocess_pipe._RESULT, [1])

        with mock.patch.object(
            _subprocess_pipe.time,
            "monotonic",
            side_effect=[10.0, 10.0, 10.6],
        ):
            self.assertIsNone(
                _subprocess_pipe._pop_buffered_serialization_error(out_q, 0.5)
            )

        out_q.get.assert_called_once_with(timeout=0.5)

    def test_buffered_error_scan_tolerates_queue_teardown(self) -> None:
        """Best-effort detail lookup cannot mask failure during queue teardown."""
        for error in (EOFError(), OSError(), RuntimeError(), ValueError()):
            with self.subTest(error=type(error).__name__):
                out_q = mock.Mock()
                out_q.get.side_effect = error

                self.assertIsNone(
                    _subprocess_pipe._pop_buffered_serialization_error(out_q, 1.0)
                )

    def test_duplicate_boundary_during_serialization_failure_waits_for_detail(
        self,
    ) -> None:
        """A buffered extra marker cannot replace the detailed feeder error."""

        async def _scenario() -> None:
            out_q: _queue.Queue[Any] = _queue.Queue()
            out_q.put((_subprocess_pipe._EPOCH_DONE, (0, 0)))
            out_q.put((_subprocess_pipe._EPOCH_DONE, (0, 0)))
            out_q.put((_subprocess_pipe._ERROR, RuntimeError("late feeder error")))
            serialization_failed = _serialization_failure_state(failed=True)
            epoch_barrier = asyncio.Event()
            with ThreadPoolExecutor(max_workers=2) as ex:
                with self.assertRaisesRegex(RuntimeError, "late feeder error"):
                    await _subprocess_pipe._collect_continuous(
                        out_q,
                        1,
                        _unbounded_queue("output"),
                        ex,
                        epoch_barrier,
                        asyncio.Event(),
                        (("output", serialization_failed),),
                    )
            self.assertFalse(epoch_barrier.is_set())

        asyncio.run(_scenario())

    def test_continuous_collect_rejects_stale_epoch_boundary(self) -> None:
        """A stale marker cannot be mistaken for the next epoch's boundary."""

        async def _scenario() -> None:
            out_q: _queue.Queue[Any] = _queue.Queue()
            out_q.put((_subprocess_pipe._EPOCH_DONE, (0, 0)))
            out_q.put((_subprocess_pipe._EPOCH_DONE, (0, 0)))
            with ThreadPoolExecutor(max_workers=1) as ex:
                with self.assertRaisesRegex(RuntimeError, "unexpected epoch"):
                    await _subprocess_pipe._collect_continuous(
                        out_q,
                        1,
                        _unbounded_queue("output"),
                        ex,
                        asyncio.Event(),
                        asyncio.Event(),
                    )

        asyncio.run(_scenario())

    def test_continuous_collect_rejects_duplicate_worker_boundary(self) -> None:
        """One worker cannot satisfy another worker's slot in the epoch barrier."""

        async def _scenario() -> None:
            out_q: _queue.Queue[Any] = _queue.Queue()
            out_q.put((_subprocess_pipe._EPOCH_DONE, (0, 0)))
            out_q.put((_subprocess_pipe._EPOCH_DONE, (0, 0)))
            with ThreadPoolExecutor(max_workers=1) as ex:
                with self.assertRaisesRegex(RuntimeError, "more than one boundary"):
                    await _subprocess_pipe._collect_continuous(
                        out_q,
                        2,
                        _unbounded_queue("output"),
                        ex,
                        asyncio.Event(),
                        asyncio.Event(),
                    )

        asyncio.run(_scenario())


class SerializationRelayProgressTest(unittest.TestCase):
    """Serialization relay deadlines remain bounded while results keep arriving."""

    def test_boundary_selectors_distinguish_pending_and_dropped_relays(self) -> None:
        """Relay boundary selectors separate dropped detail from detail in flight."""
        input_failure = _serialization_failure_state(failed=True, relay_failed=True)
        output_failure = _serialization_failure_state(failed=True)
        failures = (("input", input_failure), ("output", output_failure))

        self.assertEqual(
            _subprocess_pipe._serialization_failure_boundaries(failures),
            "input and output",
        )
        self.assertEqual(
            _subprocess_pipe._serialization_relay_failure_boundaries(failures),
            "input",
        )
        self.assertEqual(
            _subprocess_pipe._serialization_pending_relay_boundaries(failures),
            "output",
        )

    def test_collect_skips_relay_check_without_failure(self) -> None:
        """Healthy non-continuous results avoid the asynchronous relay checker."""

        async def _scenario() -> list[Any]:
            out_q: _queue.Queue[Any] = _queue.Queue()
            out_q.put((_subprocess_pipe._RESULT, [1]))
            out_q.put((_subprocess_pipe._DONE, None))
            output_queue = _unbounded_queue("output")
            with (
                ThreadPoolExecutor(max_workers=1) as ex,
                mock.patch.object(
                    _subprocess_pipe,
                    "_check_serialization_error_relay_after_progress",
                    side_effect=AssertionError("relay checker was awaited"),
                ),
            ):
                await _subprocess_pipe._collect(
                    out_q,
                    1,
                    output_queue,
                    ex,
                    asyncio.Event(),
                    asyncio.Event(),
                )
            return [output_queue.get_nowait() for _ in range(output_queue.qsize())]

        self.assertEqual(asyncio.run(_scenario()), [1])

    def test_continuous_collect_skips_relay_check_without_failure(self) -> None:
        """Healthy continuous results avoid the asynchronous relay checker."""

        async def _scenario() -> tuple[list[Any], bool]:
            out_q: _queue.Queue[Any] = _queue.Queue()
            out_q.put((_subprocess_pipe._RESULT, [1]))
            out_q.put((_subprocess_pipe._EPOCH_DONE, (0, 0)))
            out_q.put((_subprocess_pipe._DONE, None))
            output_queue = _unbounded_queue("output")
            epoch_barrier = asyncio.Event()
            with (
                ThreadPoolExecutor(max_workers=1) as ex,
                mock.patch.object(
                    _subprocess_pipe,
                    "_check_serialization_error_relay_after_progress",
                    side_effect=AssertionError("relay checker was awaited"),
                ),
            ):
                await _subprocess_pipe._collect_continuous(
                    out_q,
                    1,
                    output_queue,
                    ex,
                    epoch_barrier,
                    asyncio.Event(),
                )
            return (
                [output_queue.get_nowait() for _ in range(output_queue.qsize())],
                epoch_barrier.is_set(),
            )

        self.assertEqual(asyncio.run(_scenario()), ([1, _EPOCH_END], True))

    def test_mixed_relay_failure_waits_for_pending_detailed_error(self) -> None:
        """One generic failure cannot mask another boundary's detailed error."""

        async def _scenario(non_payload_failed: bool) -> tuple[int, bool]:
            out_q = _ScriptedResultQueue(
                [
                    (_subprocess_pipe._RESULT, [1]),
                    (_subprocess_pipe._ERROR, RuntimeError("output detail")),
                    (_subprocess_pipe._DONE, None),
                ]
            )
            input_failure = _serialization_failure_state(
                failed=True,
                relay_failed=True,
                non_payload_failed=non_payload_failed,
            )
            output_failure = _serialization_failure_state(failed=True)
            abort = asyncio.Event()
            with ThreadPoolExecutor(max_workers=1) as ex:
                with self.assertRaisesRegex(RuntimeError, "output detail"):
                    await _subprocess_pipe._collect(
                        out_q,
                        1,
                        _unbounded_queue("output"),
                        ex,
                        abort,
                        asyncio.Event(),
                        (
                            ("input", input_failure),
                            ("output", output_failure),
                        ),
                    )
            return out_q.calls, abort.is_set()

        for non_payload_failed in (False, True):
            with self.subTest(non_payload_failed=non_payload_failed):
                self.assertEqual(
                    asyncio.run(_scenario(non_payload_failed)),
                    (3, True),
                )

    def test_collect_checks_deadline_during_busy_result_stream(self) -> None:
        """Non-continuous collection checks the relay deadline before each result."""

        async def _scenario() -> int:
            failure = _serialization_failure_state(failed=True)
            out_q = _BusyResultQueue()
            with (
                ThreadPoolExecutor(max_workers=1) as ex,
                mock.patch.object(
                    _subprocess_pipe, "_SERIALIZATION_ERROR_RELAY_TIMEOUT", 0.0
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "Fused subprocess output serialization failed"
                ):
                    await _subprocess_pipe._collect(
                        out_q,
                        1,
                        _unbounded_queue("output"),
                        ex,
                        asyncio.Event(),
                        asyncio.Event(),
                        (("output", failure),),
                    )
            return out_q.calls

        self.assertEqual(asyncio.run(_scenario()), 3)

    def test_continuous_collect_checks_deadline_during_busy_stream(self) -> None:
        """Continuous collection checks the relay deadline before each result."""

        async def _scenario() -> tuple[int, bool]:
            failure = _serialization_failure_state(failed=True)
            out_q = _BusyResultQueue()
            epoch_barrier = asyncio.Event()
            with (
                ThreadPoolExecutor(max_workers=1) as ex,
                mock.patch.object(
                    _subprocess_pipe, "_SERIALIZATION_ERROR_RELAY_TIMEOUT", 0.0
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "Fused subprocess output serialization failed"
                ):
                    await _subprocess_pipe._collect_continuous(
                        out_q,
                        1,
                        _unbounded_queue("output"),
                        ex,
                        epoch_barrier,
                        asyncio.Event(),
                        (("output", failure),),
                    )
            return out_q.calls, epoch_barrier.is_set()

        self.assertEqual(asyncio.run(_scenario()), (3, False))

    def test_dropped_detailed_error_skips_relay_grace(self) -> None:
        """A known-full fallback queue fails without waiting for the deadline."""

        async def _scenario() -> None:
            failure = _serialization_failure_state(failed=True, relay_failed=True)
            out_q = _BusyResultQueue()
            with (
                ThreadPoolExecutor(max_workers=1) as ex,
                mock.patch.object(
                    _subprocess_pipe,
                    "_check_serialization_error_relay",
                    side_effect=AssertionError("relay grace was consulted"),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "Fused subprocess output serialization failed"
                ):
                    await _subprocess_pipe._collect(
                        out_q,
                        1,
                        _unbounded_queue("output"),
                        ex,
                        asyncio.Event(),
                        asyncio.Event(),
                        (("output", failure),),
                    )

        asyncio.run(_scenario())


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

    def test_drain_one_reports_queue_teardown(self) -> None:
        """A closed worker queue fails promptly instead of looking temporarily empty."""
        for error in (EOFError(), OSError(), RuntimeError(), ValueError()):
            with self.subTest(error=type(error).__name__):
                out_q = mock.Mock()
                out_q.get.side_effect = error

                with self.assertRaisesRegex(
                    RuntimeError, "worker output queue could not be read"
                ) as raised:
                    _subprocess_pipe._drain_one(out_q)

                self.assertIs(raised.exception.__cause__, error)

    def test_queue_teardown_reports_known_serialization_failure(self) -> None:
        """A known feeder failure takes precedence when its detail queue closes."""

        async def _scenario(continuous: bool, queue_error: BaseException) -> None:
            out_q = mock.Mock()
            out_q.get.side_effect = queue_error
            failure = _serialization_failure_state(failed=True)
            with ThreadPoolExecutor(max_workers=1) as ex:
                if continuous:
                    await _subprocess_pipe._collect_continuous(
                        out_q,
                        1,
                        _unbounded_queue("output"),
                        ex,
                        asyncio.Event(),
                        asyncio.Event(),
                        (("output", failure),),
                    )
                else:
                    await _subprocess_pipe._collect(
                        out_q,
                        1,
                        _unbounded_queue("output"),
                        ex,
                        asyncio.Event(),
                        asyncio.Event(),
                        (("output", failure),),
                    )

        for continuous in (False, True):
            for queue_error in (
                RuntimeError("interpreter queue is closed"),
                ValueError("output queue is closed"),
            ):
                with self.subTest(
                    continuous=continuous,
                    error=type(queue_error).__name__,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "Fused subprocess output serialization failed"
                    ):
                        asyncio.run(_scenario(continuous, queue_error))

    def test_collect_suppresses_stall_during_serialization_relay_grace(self) -> None:
        """Relay grace, not the worker-stall guard, surfaces feeder detail."""

        async def _scenario() -> int:
            out_q = _ScriptedResultQueue(
                [
                    None,
                    (_subprocess_pipe._ERROR, RuntimeError("delayed detail")),
                    (_subprocess_pipe._DONE, None),
                ]
            )
            failure = _serialization_failure_state(failed=True)
            with (
                ThreadPoolExecutor(max_workers=1) as ex,
                mock.patch.object(_subprocess_pipe, "_WORKER_STALL_TIMEOUT", -1.0),
            ):
                with self.assertRaisesRegex(RuntimeError, "delayed detail"):
                    await _subprocess_pipe._collect(
                        out_q,
                        1,
                        _unbounded_queue("output"),
                        ex,
                        asyncio.Event(),
                        asyncio.Event(),
                        (("output", failure),),
                    )
            return out_q.calls

        self.assertEqual(asyncio.run(_scenario()), 3)

    def test_continuous_collect_suppresses_stall_during_relay_grace(self) -> None:
        """Continuous relay grace cannot be preempted by the stall guard."""

        async def _scenario() -> int:
            out_q = _ScriptedResultQueue(
                [None, (_subprocess_pipe._ERROR, RuntimeError("delayed detail"))]
            )
            failure = _serialization_failure_state(failed=True)
            with (
                ThreadPoolExecutor(max_workers=1) as ex,
                mock.patch.object(_subprocess_pipe, "_WORKER_STALL_TIMEOUT", -1.0),
            ):
                with self.assertRaisesRegex(RuntimeError, "delayed detail"):
                    await _subprocess_pipe._collect_continuous(
                        out_q,
                        1,
                        _unbounded_queue("output"),
                        ex,
                        asyncio.Event(),
                        asyncio.Event(),
                        (("output", failure),),
                    )
            return out_q.calls

        self.assertEqual(asyncio.run(_scenario()), 2)

    def test_collect_suppresses_stall_while_feeder_idle(self) -> None:
        """An idle feeder suppresses the collector's stall guard during input starvation.

        With the timeout pinned to zero, any stall check on an empty queue would trip
        instantly; the collector must instead keep draining while ``feeder_idle`` is set
        (nothing dispatched, no worker message due) and still finish once the worker reports
        ``_DONE``.
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
