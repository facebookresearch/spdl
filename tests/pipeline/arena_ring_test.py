# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, cast

from spdl.pipeline._arena import SharedMemoryRingBuffer
from spdl.pipeline._arena._offload import _offload, _restore
from spdl.pipeline._arena._registry import _default_registry


class SharedMemoryRingBufferTest(unittest.TestCase):
    def _ring(self, capacity: int) -> SharedMemoryRingBuffer:
        buf = SharedMemoryRingBuffer(capacity=capacity)
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
        """A write larger than the free space is refused, not truncated."""
        buf = self._ring(64)
        writer = buf.open_writer()
        writer.begin_unit()
        with self.assertRaises(BufferError):
            writer.write_binary(b"x" * 128)


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
