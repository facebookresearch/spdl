# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from multiprocessing import shared_memory
from unittest.mock import Mock, patch

from spdl.pipeline._arena._pool import SharedMemorySegmentPool
from spdl.pipeline._arena._ring import SharedMemoryRingBuffer


class ArenaAllocationResetTest(unittest.TestCase):
    def test_partial_pool_allocation_unlinks_all_created_segments(self) -> None:
        """Failure creating a later segment releases the control and earlier data segments."""
        real = shared_memory.SharedMemory
        created = []

        def allocate(*args, **kwargs):
            if len(created) == 2:
                raise OSError("allocation root")
            segment = real(*args, **kwargs)
            created.append(segment)
            return segment

        with patch(
            "spdl.pipeline._arena._pool.shared_memory.SharedMemory",
            side_effect=allocate,
        ):
            with self.assertRaisesRegex(OSError, "allocation root"):
                SharedMemorySegmentPool(64, 3)
        self.assertEqual(len(created), 2)
        for segment in created:
            with self.assertRaises(FileNotFoundError):
                real(name=segment.name)
            self.assertIsNone(segment._mmap)

    def test_allocation_rollback_preserves_root_and_attempts_every_cleanup(
        self,
    ) -> None:
        """Secondary cleanup failures are notes and do not prevent remaining cleanup."""
        ctrl, data = Mock(), Mock()
        ctrl.buf = memoryview(bytearray(17))
        ctrl.unlink.side_effect = OSError("unlink secondary")
        ctrl.close.side_effect = OSError("close secondary")
        with patch(
            "spdl.pipeline._arena._pool.shared_memory.SharedMemory",
            side_effect=[ctrl, data, OSError("allocation root")],
        ):
            with self.assertRaisesRegex(OSError, "allocation root") as caught:
                SharedMemorySegmentPool(64, 2)
        self.assertEqual(len(caught.exception.__notes__), 2)
        ctrl.unlink.assert_called_once()
        ctrl.close.assert_called_once()
        data.unlink.assert_called_once()
        data.close.assert_called_once()

    def test_unread_spans_survive_reset_and_cannot_reclaim_next_epoch(self) -> None:
        """Old multi-binary descriptors are discarded before reuse, even across reset."""
        ring = SharedMemoryRingBuffer(32, acquire_timeout=0)
        self.addCleanup(ring.unlink)
        self.addCleanup(ring.close)
        writer, reader = ring.open_writer(), ring.open_reader()
        for epoch in range(8):
            writer.begin_unit()
            writer.write_binary(b"a" * 9)
            writer.write_binary(b"b" * 7)
            abandoned_span = writer.commit_unit()
            old_head, old_tail = ring._head, ring._tail
            writer.reset()
            self.assertEqual((ring._head, ring._tail), (old_head, old_tail))
            reader.end_unit(abandoned_span, [])
            self.assertEqual(ring._tail, old_head)
            writer.begin_unit()
            offset, size = writer.write_binary(bytes([epoch]) * 32)
            span = writer.commit_unit()
            self.assertEqual(
                bytes(reader.read_binary(offset, size)), bytes([epoch]) * 32
            )
            reader.end_unit(span, [])
            writer.reset()
            self.assertEqual((ring._head, ring._tail), (0, 0))

    def test_aborted_unpublished_span_is_not_counted_after_reset(self) -> None:
        """Reset drops unpublished reservation without advancing either shared watermark."""
        ring = SharedMemoryRingBuffer(32, acquire_timeout=0)
        self.addCleanup(ring.unlink)
        self.addCleanup(ring.close)
        writer = ring.open_writer()
        writer.begin_unit()
        writer.write_binary(b"a" * 16)
        writer.commit_unit()
        writer.begin_unit()
        writer.write_binary(b"unpublished")
        writer.abort_unit()
        writer.reset()
        self.assertEqual((ring._head, ring._tail), (16, 0))
        ring.open_reader().end_unit(16, [])
        writer.reset()
        self.assertEqual((ring._head, ring._tail), (0, 0))
