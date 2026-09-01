# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for ``Pipeline._get_item_nowait``, the non-blocking sink read.

The contract is three-way and has to hold identically on both sink backends: return an item,
raise ``queue.Empty`` for "not yet", raise ``EOFError`` for "never". The load-bearing property
is that polling can never *lose* an item -- that is precisely what makes this different from
``get_item(timeout=0)``, which submits a ``queue.get()`` onto the event loop and abandons the
future when the timeout expires, stranding whatever that get later receives.
"""

import queue
import threading
import time
import unittest
from typing import Any

from parameterized import parameterized  # pyre-ignore[21]
from spdl.pipeline import Pipeline, PipelineBuilder

_TIMEOUT: float = 60.0

_BACKENDS: list[tuple[str, bool]] = [("async_queue", False), ("thread_queue", True)]


def add_one(x: int) -> int:
    return x + 1


# Held by the test so an op can be pinned mid-flight without relying on a sleep: the op
# cannot return until the test releases it, so "the sink is empty" is a fact rather than a
# race the host's load could lose.
_RELEASE_OP: threading.Event = threading.Event()


def blocks_until_released(x: int) -> int:
    _RELEASE_OP.wait()
    return x


def _build(
    n: int, *, thread_queue: bool, continuous: bool = False, op: Any = add_one
) -> Pipeline[Any]:
    return (
        PipelineBuilder()
        .add_source(range(n), continuous=continuous)
        .pipe(op)
        .add_sink(n or 1)
        .build(num_threads=2, use_thread_output_queue=thread_queue)
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

        This is the regression guard for the abandoned-future bug that
        ``get_item(timeout=0)`` would have had: a dropped item leaves the drain spinning on
        ``queue.Empty`` until it gives up.
        """
        n = 200
        pipeline = _build(n, thread_queue=thread_queue)
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
