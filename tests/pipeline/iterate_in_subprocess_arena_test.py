# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from collections.abc import Iterable, Iterator

from spdl.pipeline import iterate_in_subprocess, SharedMemoryRingBuffer


def _make_items() -> list[dict[str, bytes | int]]:
    return [{"i": i, "blob": bytes([i % 256]) * 50_000} for i in range(8)]


def _failing_iterable() -> Iterator[dict[str, bytes | int]]:
    yield {"i": 0, "blob": b"x" * 50_000}
    raise ValueError("boom")


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
