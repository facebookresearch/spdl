# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ``Pipeline._get_item_nowait``, the non-blocking sink read.

The contract is three-way and has to hold identically on both sink backends: return an item,
raise ``queue.Empty`` for "not yet", raise ``EOFError`` for "never". The load-bearing property
is that polling can never *lose* an item. ``get_item(timeout=0)`` shares this one-shot polling
primitive and translates ``queue.Empty`` into ``TimeoutError``.
"""

import asyncio
import queue
import threading
import time
import unittest
from typing import Any, cast
from unittest import mock

from parameterized import parameterized  # pyre-ignore[21]
from spdl.pipeline import AsyncQueue, Pipeline, PipelineBuilder, StatsQueue
from spdl.pipeline._components._queue import _AsyncQueueWithSyncMirror

_TIMEOUT: float = 60.0

_BACKENDS: list[tuple[str, bool]] = [("async_queue", False), ("thread_queue", True)]


def add_one(x: int) -> int:
    return x + 1


# Held by the test so an op can be pinned mid-flight without relying on a sleep: the op
# cannot return until the test releases it, so "the sink is empty" is a fact rather than a
# race the host's load could lose.
_RELEASE_OP: threading.Event = threading.Event()
_EVENT_LOOP_BLOCKED: threading.Event = threading.Event()
_RELEASE_EVENT_LOOP: threading.Event = threading.Event()
_BLOCK_SCHEDULED_QUEUES: set[int] = set()


def blocks_until_released(x: int) -> int:
    _RELEASE_OP.wait()
    return x


def _block_event_loop() -> None:
    _EVENT_LOOP_BLOCKED.set()
    if not _RELEASE_EVENT_LOOP.wait(_TIMEOUT):
        raise RuntimeError("Test did not release the event loop.")


_STATS_QUEUE_PUT = StatsQueue.put


async def _put_and_block_after_first_sink_item(self: StatsQueue, item: object) -> None:
    """Pin the loop after a built-in sink queue publishes its first item."""
    await _STATS_QUEUE_PUT(self, item)
    queue_id = id(self)
    if self.info.stage_name == "sink" and queue_id not in _BLOCK_SCHEDULED_QUEUES:
        _BLOCK_SCHEDULED_QUEUES.add(queue_id)
        asyncio.get_running_loop().call_soon(_block_event_loop)


class _TransformingSinkQueue(AsyncQueue):
    """Transform values on both sides of a custom sink queue."""

    async def put(self, item: object) -> None:
        if self.info.stage_name == "sink" and isinstance(item, int):
            item += 100
        await super().put(item)

    async def get(self) -> object:
        item = await super().get()
        if self.info.stage_name == "sink" and isinstance(item, int):
            return item * 2
        return item


class _RaisingSinkGetQueue(AsyncQueue):
    """Raise from the custom sink consumption path."""

    async def get(self) -> object:
        item = await super().get()
        if self.info.stage_name == "sink" and isinstance(item, int):
            raise RuntimeError("custom sink get failure")
        return item


def _build(
    n: int,
    *,
    thread_queue: bool,
    continuous: bool = False,
    op: Any = add_one,
    queue_class: type[AsyncQueue] | None = None,
) -> Pipeline[Any]:
    return (
        PipelineBuilder()
        .add_source(range(n), continuous=continuous)
        .pipe(op)
        .add_sink(n or 1)
        .build(
            num_threads=2,
            use_thread_output_queue=thread_queue,
            queue_class=queue_class,
        )
    )


def _drain_nowait(pipeline: Pipeline[Any]) -> list[Any]:
    """Pull one stream's worth of items using *only* ``_get_item_nowait``.

    Spins on ``queue.Empty`` rather than blocking, so any item the poll drops would show up as a
    short result rather than as a hang.
    """
    out: list[Any] = []
    deadline = time.monotonic() + _TIMEOUT
    while True:
        try:
            out.append(pipeline._get_item_nowait())
        except EOFError:
            return out
        except queue.Empty:
            if time.monotonic() > deadline:
                raise AssertionError(
                    f"timed out with {len(out)} items; an item was likely dropped"
                ) from None
            time.sleep(0.001)


class GetItemNowaitTest(unittest.TestCase):
    """The three-way contract holds identically on both sink backends."""

    @parameterized.expand(_BACKENDS)
    def test_drains_every_item(self, _name: str, thread_queue: bool) -> None:
        """Polling alone yields the whole stream -- no item is stranded by a poll.

        A dropped item leaves the drain spinning on ``queue.Empty`` until it gives up.
        """
        n = 200
        pipeline = _build(n, thread_queue=thread_queue)
        with pipeline.auto_stop():
            self.assertEqual(sorted(_drain_nowait(pipeline)), [x + 1 for x in range(n)])

    def test_plain_async_queue_drains_every_item(self) -> None:
        """Polling a mirrored plain AsyncQueue yields the complete stream."""
        n = 200
        pipeline = _build(
            n,
            thread_queue=False,
            queue_class=AsyncQueue,
        )
        with pipeline.auto_stop():
            self.assertEqual(sorted(_drain_nowait(pipeline)), [x + 1 for x in range(n)])

    @parameterized.expand(_BACKENDS)
    def test_empty_when_nothing_ready(self, _name: str, thread_queue: bool) -> None:
        """A running pipeline with nothing produced yet raises ``queue.Empty``, not EOF.

        The only op is pinned inside ``_RELEASE_OP`` until this test lets it go, so the sink is
        deterministically empty while the pipeline is still running -- no sleep, and nothing for
        a loaded host to race.
        """
        _RELEASE_OP.clear()
        pipeline = _build(1, thread_queue=thread_queue, op=blocks_until_released)
        try:
            with pipeline.auto_stop():
                with self.assertRaises(queue.Empty):
                    pipeline._get_item_nowait()
                # Release before teardown, or ``stop()`` waits on the pinned op.
                _RELEASE_OP.set()
        finally:
            _RELEASE_OP.set()

    @parameterized.expand(_BACKENDS)
    def test_eof_when_exhausted(self, _name: str, thread_queue: bool) -> None:
        """Once the source is drained and the task is done, polling reports ``EOFError``."""
        pipeline = _build(4, thread_queue=thread_queue)
        with pipeline.auto_stop():
            self.assertEqual(sorted(_drain_nowait(pipeline)), [1, 2, 3, 4])
            # _drain_nowait returns on the first EOFError; nothing further can appear.
            with self.assertRaises(EOFError):
                pipeline._get_item_nowait()

    @parameterized.expand(_BACKENDS)
    def test_epoch_boundary_reported_as_eof(
        self, _name: str, thread_queue: bool
    ) -> None:
        """With a continuous source, each epoch ends in ``EOFError`` and the next one resumes.

        Mirrors how the blocking ``get_item`` reports an epoch boundary, so a caller can drain
        epoch by epoch without ever blocking.
        """
        n = 32
        ref = [x + 1 for x in range(n)]
        pipeline = _build(n, thread_queue=thread_queue, continuous=True)
        with pipeline.auto_stop():
            for epoch in range(3):
                with self.subTest(epoch=epoch):
                    self.assertEqual(sorted(_drain_nowait(pipeline)), ref)

    @parameterized.expand(_BACKENDS)
    def test_requires_started_pipeline(self, _name: str, thread_queue: bool) -> None:
        """Polling does not auto-start the pipeline the way ``get_item`` does.

        A caller polling a pipeline it never started wants that surfaced, not silently fixed.
        """
        pipeline = _build(4, thread_queue=thread_queue)
        with self.assertRaisesRegex(RuntimeError, "not started"):
            pipeline._get_item_nowait()

    def test_async_sink_poll_does_not_wait_for_busy_event_loop(self) -> None:
        """An empty poll returns promptly without stealing a later item."""
        _EVENT_LOOP_BLOCKED.clear()
        _RELEASE_EVENT_LOOP.clear()
        _BLOCK_SCHEDULED_QUEUES.clear()
        pipeline = _build(2, thread_queue=False, queue_class=StatsQueue)
        poll_thread: threading.Thread | None = None
        try:
            with mock.patch.object(
                StatsQueue, "put", _put_and_block_after_first_sink_item
            ):
                pipeline.start(timeout=_TIMEOUT)
                self.assertTrue(
                    _EVENT_LOOP_BLOCKED.wait(_TIMEOUT),
                    "event loop did not reach the deterministic blocking callback",
                )

                # The first item is already in the thread-safe mirror. Reading it must
                # not wait for the event loop callback that releases async backpressure.
                result: queue.Queue[object] = queue.Queue()
                poll_thread = threading.Thread(
                    target=lambda: result.put(_poll_nowait(pipeline))
                )
                poll_thread.start()
                poll_thread.join(timeout=1)
                self.assertFalse(
                    poll_thread.is_alive(),
                    "get_item_nowait blocked on the busy event loop",
                )
                self.assertEqual(result.get_nowait(), 1)

                # No item can be produced while the loop is pinned. This poll must
                # return Empty without leaving any loop callback that can steal item 2.
                poll_thread = threading.Thread(
                    target=lambda: result.put(_poll_nowait(pipeline))
                )
                poll_thread.start()
                poll_thread.join(timeout=1)
                self.assertFalse(
                    poll_thread.is_alive(),
                    "empty get_item_nowait blocked on the busy event loop",
                )
                self.assertIsInstance(result.get_nowait(), queue.Empty)

                _RELEASE_EVENT_LOOP.set()
                self.assertEqual(pipeline.get_item(timeout=_TIMEOUT), 2)
        finally:
            _RELEASE_EVENT_LOOP.set()
            if poll_thread is not None:
                poll_thread.join(timeout=_TIMEOUT)
            pipeline.stop(timeout=_TIMEOUT)

    def test_closed_loop_during_release_preserves_mirrored_item(self) -> None:
        """A release callback failure cannot discard an already-mirrored item."""
        _EVENT_LOOP_BLOCKED.clear()
        _RELEASE_EVENT_LOOP.clear()
        _BLOCK_SCHEDULED_QUEUES.clear()
        pipeline = _build(1, thread_queue=False, queue_class=StatsQueue)
        try:
            with mock.patch.object(
                StatsQueue, "put", _put_and_block_after_first_sink_item
            ):
                pipeline.start(timeout=_TIMEOUT)
                self.assertTrue(
                    _EVENT_LOOP_BLOCKED.wait(_TIMEOUT),
                    "event loop did not publish the mirrored item",
                )
                event_loop = pipeline._impl._event_loop
                with (
                    mock.patch.object(
                        event_loop,
                        "call_soon_threadsafe",
                        side_effect=RuntimeError("Event loop is closed"),
                    ),
                    mock.patch.object(
                        event_loop,
                        "is_task_completed",
                        return_value=False,
                    ),
                    self.assertLogs("spdl.pipeline._pipeline", level="WARNING"),
                ):
                    self.assertEqual(pipeline.get_item(timeout=0), 1)
                output_queue = pipeline._impl._output_queue
                self.assertIsInstance(output_queue, _AsyncQueueWithSyncMirror)
                assert isinstance(output_queue, _AsyncQueueWithSyncMirror)
                self.assertFalse(output_queue.full())
        finally:
            _RELEASE_EVENT_LOOP.set()
            pipeline.stop(timeout=_TIMEOUT)

    def test_custom_sink_queue_transformations_are_preserved(self) -> None:
        """Custom sink put/get transformations determine foreground values."""
        pipeline = _build(
            1,
            thread_queue=False,
            continuous=True,
            queue_class=_TransformingSinkQueue,
        )

        with pipeline.auto_stop(timeout=_TIMEOUT):
            self.assertEqual(pipeline.get_item(timeout=_TIMEOUT), 202)

    def test_completed_custom_sink_queue_transformations_are_preserved(self) -> None:
        """A completed pipeline still reads custom sinks through their async get."""
        pipeline = (
            PipelineBuilder()
            .add_source([1])
            .pipe(add_one)
            .add_sink(2)
            .build(
                num_threads=2,
                use_thread_output_queue=False,
                queue_class=_TransformingSinkQueue,
            )
        )

        with pipeline.auto_stop(timeout=_TIMEOUT):
            self.assertTrue(pipeline._impl._event_loop._task_completed.wait(_TIMEOUT))
            self.assertEqual(pipeline.get_item(timeout=_TIMEOUT), 204)
            with self.assertRaises(EOFError):
                pipeline.get_item(timeout=_TIMEOUT)

    def test_stopped_async_sink_drains_buffered_item(self) -> None:
        """A stopped owner loop leaves buffered async-sink output readable."""
        pipeline = (
            PipelineBuilder()
            .add_source([1])
            .pipe(add_one)
            .add_sink(2)
            .build(
                num_threads=2,
                use_thread_output_queue=False,
                queue_class=AsyncQueue,
            )
        )

        try:
            pipeline.start(timeout=_TIMEOUT)
            self.assertTrue(pipeline._impl._event_loop._task_completed.wait(_TIMEOUT))
            pipeline.stop(timeout=_TIMEOUT)
            self.assertEqual(pipeline.get_item(timeout=_TIMEOUT), 2)
            with self.assertRaises(EOFError):
                pipeline.get_item(timeout=_TIMEOUT)
        finally:
            pipeline.stop(timeout=_TIMEOUT)

    def test_custom_sink_queue_get_exceptions_are_preserved(self) -> None:
        """Custom sink get failures propagate to the foreground caller."""
        pipeline = _build(
            1,
            thread_queue=False,
            continuous=True,
            queue_class=_RaisingSinkGetQueue,
        )

        with pipeline.auto_stop(timeout=_TIMEOUT):
            with self.assertRaisesRegex(
                RuntimeError,
                "custom sink get failure",
            ):
                pipeline.get_item(timeout=_TIMEOUT)

    def test_zero_timeout_surfaces_failure_racing_empty_poll(self) -> None:
        """A zero-timeout read surfaces a task failure racing an empty mirror."""
        _RELEASE_OP.clear()
        pipeline = _build(
            1,
            thread_queue=False,
            op=blocks_until_released,
            queue_class=StatsQueue,
        )
        try:
            pipeline.start(timeout=_TIMEOUT)
            event_loop = pipeline._impl._event_loop
            with (
                mock.patch.object(
                    event_loop,
                    "is_task_completed",
                    side_effect=[False, True],
                ),
                mock.patch.object(
                    event_loop,
                    "observe_task_exception",
                    return_value=RuntimeError("task failed during poll"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "task failed during poll"):
                    pipeline.get_item(timeout=0)
        finally:
            _RELEASE_OP.set()
            pipeline.stop(timeout=_TIMEOUT)

    def test_zero_timeout_waits_for_in_flight_mirror_publication(self) -> None:
        """An unpublished mirror item cannot be mistaken for completed EOF."""
        publication_started = threading.Event()
        release_publication = threading.Event()
        pipeline = _build(1, thread_queue=False, queue_class=StatsQueue)
        output_queue = cast(
            _AsyncQueueWithSyncMirror,
            pipeline._impl._output_queue,
        )
        publish = output_queue._sync_queue.put_nowait

        def publish_after_release(item: object) -> None:
            publication_started.set()
            if not release_publication.wait(_TIMEOUT):
                raise RuntimeError("Test did not release mirror publication.")
            publish(item)

        try:
            with mock.patch.object(
                output_queue._sync_queue,
                "put_nowait",
                side_effect=publish_after_release,
            ):
                pipeline.start(timeout=_TIMEOUT)
                self.assertTrue(publication_started.wait(_TIMEOUT))
                self.assertFalse(pipeline._impl._event_loop.is_task_completed())

                with self.assertRaises(TimeoutError):
                    pipeline.get_item(timeout=0)

                release_publication.set()
                self.assertEqual(pipeline.get_item(timeout=_TIMEOUT), 1)
        finally:
            release_publication.set()
            pipeline.stop(timeout=_TIMEOUT)


def _poll_nowait(pipeline: Pipeline[Any]) -> object:
    """Return either a polled item or its exception for a helper thread."""
    try:
        return pipeline._get_item_nowait()
    except Exception as exc:
        return exc
