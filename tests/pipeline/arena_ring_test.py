# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import threading
import time
import unittest
from typing import Any, cast

from spdl.pipeline._arena import SharedMemoryRingBuffer
from spdl.pipeline._arena._offload import _offload, _restore
from spdl.pipeline._arena._registry import _default_registry


class SharedMemoryRingBufferTest(unittest.TestCase):
    def _ring(
        self, capacity: int, acquire_timeout: float = 60.0
    ) -> SharedMemoryRingBuffer:
        buf = SharedMemoryRingBuffer(capacity=capacity, acquire_timeout=acquire_timeout)
        self.addCleanup(buf.unlink)
        self.addCleanup(buf.close)
        return buf

    def test_write_then_read_round_trips(self) -> None:
        """A binary written to the arena reads back byte-for-byte."""
        buf = self._ring(1 << 16)
        writer = buf.open_writer()
        reader = buf.open_reader()
        writer.begin_unit()
        offset, nbytes = writer.write_binary(b"hello world")
        span = writer.commit_unit()
        self.assertEqual(reader.read_binary(offset, nbytes), b"hello world")
        reader.end_unit(span, [])

    def test_reader_sees_nothing_until_unit_committed(self) -> None:
        """A unit is invisible to the reader until it is published."""
        buf = self._ring(1 << 16)
        writer = buf.open_writer()
        writer.begin_unit()
        writer.write_binary(b"x" * 100)
        # head is only advanced by commit_unit, so the reader's view is empty.
        self.assertEqual(buf.head, 0)
        writer.commit_unit()
        self.assertEqual(buf.head, 100)

    def test_release_advances_tail_in_bulk(self) -> None:
        """Releasing a unit returns its whole region in one tail advance."""
        buf = self._ring(1 << 16)
        writer = buf.open_writer()
        reader = buf.open_reader()
        writer.begin_unit()
        writer.write_binary(b"a" * 30)
        writer.write_binary(b"b" * 70)
        span = writer.commit_unit()
        self.assertEqual(span, 100)
        reader.end_unit(span, [])
        self.assertEqual(buf.tail, 100)

    def test_payload_wrapping_physical_end_is_intact(self) -> None:
        """Data that straddles the physical end of the arena is preserved."""
        buf = self._ring(64)
        writer = buf.open_writer()
        reader = buf.open_reader()
        # Consume most of the arena so the next write wraps the seam.
        writer.begin_unit()
        off0, n0 = writer.write_binary(b"p" * 50)
        writer.commit_unit()
        reader.read_binary(off0, n0)
        reader.end_unit(50, [])

        writer.begin_unit()
        offset, nbytes = writer.write_binary(b"abcdefghijklmnop")  # 16 bytes, wraps
        span = writer.commit_unit()
        self.assertEqual(reader.read_binary(offset, nbytes), b"abcdefghijklmnop")
        reader.end_unit(span, [])

    def test_overrun_raises(self) -> None:
        """A write larger than the entire ring is refused immediately, not blocked."""
        buf = self._ring(64)
        writer = buf.open_writer()
        writer.begin_unit()
        # 128 > capacity 64: no consumer reclaim could ever satisfy it, so the
        # producer must raise promptly rather than block on the cv.
        t0 = time.monotonic()
        with self.assertRaises(BufferError):
            writer.write_binary(b"x" * 128)
        self.assertLess(time.monotonic() - t0, 1.0)

    def test_unit_overflow_raises_promptly(self) -> None:
        """A unit accumulating more bytes than the ring can ever hold raises fast."""
        buf = self._ring(64)
        writer = buf.open_writer()
        writer.begin_unit()
        writer.write_binary(b"a" * 50)
        # The in-progress unit (50 + 30 = 80) exceeds capacity 64. Even with a
        # fully drained tail, committing this unit could never make those bytes
        # reclaimable, so the producer must raise instead of blocking.
        t0 = time.monotonic()
        with self.assertRaises(BufferError):
            writer.write_binary(b"b" * 30)
        self.assertLess(time.monotonic() - t0, 1.0)

    def test_nonblocking_full_raises(self) -> None:
        """With acquire_timeout=0, a write that would not currently fit raises."""
        buf = self._ring(64, acquire_timeout=0.0)
        writer = buf.open_writer()
        writer.begin_unit()
        writer.write_binary(b"x" * 50)
        writer.commit_unit()
        writer.begin_unit()
        with self.assertRaises(BufferError):
            writer.write_binary(b"y" * 50)  # 50 > free 14

    def test_blocking_then_resume(self) -> None:
        """A producer blocked on a full ring resumes when the consumer releases space."""
        buf = self._ring(64, acquire_timeout=5.0)
        writer = buf.open_writer()
        reader = buf.open_reader()

        # Fill the ring with one committed unit.
        writer.begin_unit()
        off0, n0 = writer.write_binary(b"x" * 50)
        span0 = writer.commit_unit()

        def _release_soon() -> None:
            time.sleep(0.1)
            reader.read_binary(off0, n0)
            reader.end_unit(span0, [])

        t = threading.Thread(target=_release_soon)
        t.start()
        try:
            t0 = time.monotonic()
            writer.begin_unit()
            # 50 bytes is more than the 14 free; producer blocks until release.
            off1, n1 = writer.write_binary(b"y" * 50)
            elapsed = time.monotonic() - t0
        finally:
            t.join()
        writer.commit_unit()
        # Should have unblocked roughly when the consumer released, well under
        # the 5s acquire_timeout, and not "instant" (proves it actually waited).
        self.assertLess(elapsed, 2.0)
        self.assertGreaterEqual(elapsed, 0.05)

    def test_blocking_timeout_raises(self) -> None:
        """A producer with no consumer raises BufferError after acquire_timeout."""
        buf = self._ring(64, acquire_timeout=0.05)
        writer = buf.open_writer()
        writer.begin_unit()
        writer.write_binary(b"x" * 50)
        writer.commit_unit()
        writer.begin_unit()
        t0 = time.monotonic()
        with self.assertRaises(BufferError):
            writer.write_binary(b"y" * 50)
        elapsed = time.monotonic() - t0
        self.assertGreaterEqual(elapsed, 0.04)
        self.assertLess(elapsed, 2.0)

    def test_shutdown_wakes_blocked_producer(self) -> None:
        """shutdown_arena() wakes a blocked producer immediately.

        Models the consumer-dies-first scenario: even though no reclaim ever
        happens, the parent flipping the shutdown flag must let the producer
        exit cleanly so the worker process can shut down without hanging.
        """
        buf = self._ring(64, acquire_timeout=10.0)
        writer = buf.open_writer()
        writer.begin_unit()
        writer.write_binary(b"x" * 50)
        writer.commit_unit()
        writer.begin_unit()

        def _shutdown_soon() -> None:
            time.sleep(0.1)
            buf.shutdown_arena()

        t = threading.Thread(target=_shutdown_soon)
        t.start()
        try:
            t0 = time.monotonic()
            with self.assertRaises(BufferError):
                writer.write_binary(b"y" * 50)
            elapsed = time.monotonic() - t0
        finally:
            t.join()
        self.assertLess(elapsed, 2.0)


class OffloadRestoreTest(unittest.TestCase):
    def _ring(self, capacity: int) -> SharedMemoryRingBuffer:
        buf = SharedMemoryRingBuffer(capacity=capacity)
        self.addCleanup(buf.unlink)
        self.addCleanup(buf.close)
        return buf

    def test_large_bytes_round_trip_through_arena(self) -> None:
        """A nested object with a large bytes field round-trips via the arena."""
        buf = self._ring(1 << 20)
        writer, reader = buf.open_writer(), buf.open_reader()
        registry = _default_registry()
        obj = {"id": 7, "blob": b"z" * 100_000, "tags": ["a", "b"]}
        out = _restore(_offload(obj, writer, registry), reader, registry)
        self.assertEqual(out, obj)

    def test_bytes_like_inputs_preserve_type_on_ring(self) -> None:
        """The ring copies payloads out, so bytes/bytearray/memoryview inputs each
        restore as the same type with identical contents (it is not the zero-copy
        backend, so the original type is reconstructed)."""
        base = bytes(range(256)) * 1000  # 256_000 bytes, above the threshold
        cases = [
            ("bytes", base, bytes),
            ("bytearray", bytearray(base), bytearray),
            ("memoryview", memoryview(base), memoryview),
            ("noncontiguous_memoryview", memoryview(base)[::2], memoryview),
        ]
        for label, src, expected_type in cases:
            with self.subTest(input=label):
                buf = self._ring(1 << 20)
                writer, reader = buf.open_writer(), buf.open_reader()
                registry = _default_registry()
                out = cast(
                    dict[str, Any],
                    _restore(_offload({"b": src}, writer, registry), reader, registry),
                )
                self.assertIsInstance(out["b"], expected_type)
                self.assertEqual(bytes(out["b"]), bytes(src))

    def test_small_fields_stay_inline(self) -> None:
        """Fields below the threshold are not offloaded (zero arena span)."""
        buf = self._ring(1 << 16)
        writer, reader = buf.open_writer(), buf.open_reader()
        registry = _default_registry()
        blob = _offload({"small": b"tiny", "n": 5}, writer, registry)
        # First 8 bytes encode the unit span; nothing offloaded -> span 0.
        self.assertEqual(blob[:8], b"\x00" * 8)
        self.assertEqual(_restore(blob, reader, registry), {"small": b"tiny", "n": 5})

    def test_many_units_reuse_wrapped_space(self) -> None:
        """Repeated offload/restore reuses arena space across the wrap point."""
        buf = self._ring(8192)
        writer, reader = buf.open_writer(), buf.open_reader()
        registry = _default_registry()
        for i in range(100):
            payload = bytes([i % 256]) * 5000
            obj = {"i": i, "blob": payload}
            out = _restore(_offload(obj, writer, registry), reader, registry)
            self.assertEqual(out, obj)

    def test_numpy_array_round_trips(self) -> None:
        """A NumPy array offloaded to the arena restores equal."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not available")
        buf = self._ring(1 << 20)
        writer, reader = buf.open_writer(), buf.open_reader()
        registry = _default_registry()
        arr = np.arange(10_000, dtype=np.float64).reshape(100, 100)
        out = cast(
            dict[str, Any],
            _restore(_offload({"a": arr}, writer, registry), reader, registry),
        )
        self.assertTrue(np.array_equal(out["a"], arr))
        self.assertEqual(out["a"].dtype, arr.dtype)
