# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import time
import unittest
from collections.abc import Iterable, Iterator

from spdl.pipeline import iterate_in_subprocess, SharedMemoryRingBuffer


def _make_items() -> list[dict[str, bytes | int]]:
    return [{"i": i, "blob": bytes([i % 256]) * 50_000} for i in range(8)]


def _failing_iterable() -> Iterator[dict[str, bytes | int]]:
    yield {"i": 0, "blob": b"x" * 50_000}
    raise ValueError("boom")


def _producer_blocking_then_dying() -> Iterator[dict[str, bytes | int]]:
    """Yields one large item, then dies — modeling a producer crash mid-stream."""
    yield {"i": 0, "blob": b"x" * 100_000}
    raise RuntimeError("producer died")


def _producer_blocked_on_full_arena() -> Iterator[dict[str, bytes | int]]:
    """Pushes more units than the arena/queue can hold, exercising producer block."""
    for i in range(200):
        yield {"i": i, "blob": bytes([i % 256]) * 100_000}


class IterateInSubprocessArenaTest(unittest.TestCase):
    def test_round_trip_through_arena(self) -> None:
        """Items with large bytes round-trip through the shared-memory arena."""
        buf = SharedMemoryRingBuffer(capacity=1 << 20)
        got = list(iterate_in_subprocess(_make_items, arena=buf, buffer_size=2))
        self.assertEqual(got, _make_items())

    def test_reiteration_yields_full_sequence(self) -> None:
        """Re-iterating the reused worker reclaims the arena and yields all items."""
        buf = SharedMemoryRingBuffer(capacity=1 << 20)
        it: Iterable[dict[str, bytes | int]] = iterate_in_subprocess(
            _make_items, arena=buf, buffer_size=2
        )
        self.assertEqual(list(it), _make_items())
        self.assertEqual(list(it), _make_items())

    def test_worker_error_propagates_without_hanging(self) -> None:
        """A worker exception surfaces as RuntimeError instead of deadlocking."""
        buf = SharedMemoryRingBuffer(capacity=1 << 20)
        it = iterate_in_subprocess(_failing_iterable, arena=buf, buffer_size=2)
        with self.assertRaises(RuntimeError):
            list(it)

    def test_producer_death_does_not_hang_consumer(self) -> None:
        """A worker that crashes mid-stream surfaces as an error within a bounded time."""
        buf = SharedMemoryRingBuffer(capacity=1 << 20, acquire_timeout=2.0)
        it = iterate_in_subprocess(
            _producer_blocking_then_dying, arena=buf, buffer_size=2
        )
        t0 = time.monotonic()
        with self.assertRaises(RuntimeError):
            list(it)
        # Whether we surface the producer's RuntimeError or the arena's
        # acquire timeout, the consumer must not hang.
        self.assertLess(time.monotonic() - t0, 30.0)

    def test_consumer_exit_during_producer_block_is_clean(self) -> None:
        """Closing the iterable while the producer is blocked tears down cleanly.

        The producer is generating units faster than we consume them, so it
        eventually blocks in ``write_binary`` waiting for the consumer to
        release space. Closing the iterable from the consumer side must wake
        the worker (via the arena's shutdown flag) so the process exits within
        the teardown grace period instead of hanging at interpreter exit.
        """
        buf = SharedMemoryRingBuffer(capacity=1 << 18, acquire_timeout=30.0)
        it = iterate_in_subprocess(
            _producer_blocked_on_full_arena, arena=buf, buffer_size=1
        )
        iterator = iter(it)
        # Pull just a few items so the producer fills the arena and queue, then
        # blocks in ``write_binary`` for the rest of its 200-item sequence.
        for _ in range(3):
            next(iterator)
        t0 = time.monotonic()
        # Drop the iterator and the iterable; teardown should release the
        # blocked worker and unlink the arena promptly.
        del iterator
        del it
        # Allow GC to run teardown.
        import gc as _gc

        _gc.collect()
        self.assertLess(time.monotonic() - t0, 15.0)
