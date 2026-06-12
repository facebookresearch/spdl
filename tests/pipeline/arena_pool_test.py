# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import gc
import struct
import threading
import time
import unittest
from typing import Any, cast

from spdl.pipeline import iterate_in_subprocess, SharedMemorySegmentPool
from spdl.pipeline._arena._offload import _offload, _restore
from spdl.pipeline._arena._registry import _default_registry


def _make_items() -> list[dict[str, bytes | int]]:
    return [{"i": i, "blob": bytes([i % 256]) * 40_000} for i in range(8)]


class SharedMemorySegmentPoolTest(unittest.TestCase):
    def _pool(
        self, segment_size: int, count: int, acquire_timeout: float = 60.0
    ) -> SharedMemorySegmentPool:
        pool = SharedMemorySegmentPool(
            segment_size=segment_size, count=count, acquire_timeout=acquire_timeout
        )
        self.addCleanup(pool.unlink)
        self.addCleanup(pool.close)
        return pool

    def test_unit_round_trips_through_a_segment(self) -> None:
        """A unit written to a segment reads back byte-for-byte."""
        pool = self._pool(1 << 16, 2)
        writer, reader = pool.open_writer(), pool.open_reader()
        writer.begin_unit()
        off, n = writer.write_binary(b"hello pool")
        span = writer.commit_unit()
        self.assertEqual(reader.read_binary(off, n), b"hello pool")
        reader.end_unit(span, [])

    def test_units_rotate_across_segments(self) -> None:
        """Consecutive units land in successive segments and round-trip."""
        pool = self._pool(1 << 16, 3)
        writer, reader = pool.open_writer(), pool.open_reader()
        for i in range(10):
            payload = bytes([i % 256]) * 1000
            writer.begin_unit()
            off, n = writer.write_binary(payload)
            span = writer.commit_unit()
            self.assertEqual(reader.read_binary(off, n), payload)
            reader.end_unit(span, [])

    def test_write_binary_offsets_are_aligned(self) -> None:
        """Each binary is placed on a 64-byte boundary (for zero-copy views)."""
        pool = self._pool(1 << 16, 2)
        writer = pool.open_writer()
        writer.begin_unit()
        off1, _ = writer.write_binary(b"x" * 10)
        off2, _ = writer.write_binary(b"y" * 100)
        off3, _ = writer.write_binary(b"z" * 3)
        writer.commit_unit()
        self.assertEqual(off1 % 64, 0)
        self.assertEqual(off2 % 64, 0)
        self.assertEqual(off3 % 64, 0)

    def test_close_tolerates_live_view(self) -> None:
        """close() must not raise even if a zero-copy view still maps a segment."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not available")
        pool = SharedMemorySegmentPool(segment_size=1 << 20, count=2)
        self.addCleanup(pool.unlink)
        writer, reader = pool.open_writer(), pool.open_reader()
        registry = _default_registry()
        out = _restore(
            _offload({"a": np.arange(1000, dtype=np.int64)}, writer, registry),
            reader,
            registry,
        )
        view = cast(dict[str, Any], out)["a"]
        # A live view keeps the segment exported; close() must not raise.
        pool.close()
        pool.unlink()
        # The view is still valid after close()+unlink() (the mapping persists
        # until it is released).
        self.assertEqual(int(view[0]), 0)
        del out, view
        gc.collect()

    def test_exhausted_pool_raises(self) -> None:
        """With acquire_timeout=0 (Mode A), a unit with no free segment is refused."""
        pool = self._pool(1 << 16, 1, acquire_timeout=0.0)
        writer = pool.open_writer()
        writer.begin_unit()
        writer.write_binary(b"x" * 10)
        writer.commit_unit()  # the single segment is now in flight (unreclaimed)
        with self.assertRaises(BufferError):
            writer.begin_unit()

    def test_begin_unit_blocks_until_reclaim(self) -> None:
        """Mode B: begin_unit waits for a reclaim instead of failing when full."""
        pool: SharedMemorySegmentPool = self._pool(1 << 16, 1, acquire_timeout=5.0)
        writer = pool.open_writer()
        writer.begin_unit()
        writer.write_binary(b"x" * 10)
        writer.commit_unit()  # pool full: published=1, reclaimed=0, count=1

        def _reclaim_soon() -> None:
            time.sleep(0.1)
            pool.reclaimed = pool.reclaimed + 1  # simulate the consumer releasing

        t = threading.Thread(target=_reclaim_soon)
        t.start()
        try:
            t0 = time.monotonic()
            writer.begin_unit()
            elapsed = time.monotonic() - t0
        finally:
            t.join()
        off, n = writer.write_binary(b"y" * 10)
        self.assertEqual(writer.commit_unit(), n)
        # Producer wakes promptly (cv-driven, not poll-bounded).
        self.assertLess(elapsed, 2.0)
        self.assertGreaterEqual(elapsed, 0.05)

    def test_begin_unit_timeout_raises(self) -> None:
        """begin_unit raises BufferError after acquire_timeout when no reclaim."""
        pool = self._pool(1 << 16, 1, acquire_timeout=0.05)
        writer = pool.open_writer()
        writer.begin_unit()
        writer.write_binary(b"x" * 10)
        writer.commit_unit()  # pool full
        t0 = time.monotonic()
        with self.assertRaises(BufferError):
            writer.begin_unit()
        elapsed = time.monotonic() - t0
        self.assertGreaterEqual(elapsed, 0.04)
        self.assertLess(elapsed, 2.0)

    def test_shutdown_wakes_blocked_producer(self) -> None:
        """shutdown_arena() wakes a blocked producer immediately.

        Models the consumer-dies-first scenario: with no reclaim ever happening,
        flipping the shutdown flag still lets the producer exit so the worker
        process can shut down without hanging.
        """
        pool = self._pool(1 << 16, 1, acquire_timeout=10.0)
        writer = pool.open_writer()
        writer.begin_unit()
        writer.write_binary(b"x" * 10)
        writer.commit_unit()

        def _shutdown_soon() -> None:
            time.sleep(0.1)
            pool.shutdown_arena()

        t = threading.Thread(target=_shutdown_soon)
        t.start()
        try:
            t0 = time.monotonic()
            with self.assertRaises(BufferError):
                writer.begin_unit()
            elapsed = time.monotonic() - t0
        finally:
            t.join()
        self.assertLess(elapsed, 2.0)

    def test_unit_larger_than_segment_raises(self) -> None:
        """A unit that does not fit a segment is refused, not truncated."""
        pool = self._pool(1024, 2)
        writer = pool.open_writer()
        writer.begin_unit()
        with self.assertRaises(BufferError):
            writer.write_binary(b"x" * 2048)

    def test_offload_restore_round_trips(self) -> None:
        """A nested object with a large field round-trips via the pool."""
        pool = self._pool(1 << 20, 2)
        writer, reader = pool.open_writer(), pool.open_reader()
        registry = _default_registry()
        obj = {"id": 3, "blob": b"q" * 200_000, "tags": ["x"]}
        self.assertEqual(
            _restore(_offload(obj, writer, registry), reader, registry), obj
        )

    def test_numpy_restore_is_a_zero_copy_view(self) -> None:
        """A restored NumPy array aliases the segment (no copy)."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not available")
        pool = self._pool(1 << 20, 2)
        writer, reader = pool.open_writer(), pool.open_reader()
        registry = _default_registry()
        arr = np.arange(1000, dtype=np.int64)
        out = _restore(_offload({"a": arr}, writer, registry), reader, registry)
        view = cast(dict[str, Any], out)["a"]
        self.assertTrue(np.array_equal(view, arr))
        # Mutating the segment's first int64 is observed through the view, which
        # proves the view aliases shared memory rather than a copy.
        struct.pack_into("<q", pool._segment(0), 0, 999)
        self.assertEqual(int(view[0]), 999)
        # Release the view before teardown so close() has no live exports.
        del out, view
        gc.collect()

    def test_segment_reclaimed_only_after_view_released(self) -> None:
        """The pool holds a segment until its zero-copy view is dropped."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not available")
        pool = self._pool(1 << 20, 2)
        writer, reader = pool.open_writer(), pool.open_reader()
        registry = _default_registry()
        out = _restore(
            _offload({"a": np.arange(1000, dtype=np.int64)}, writer, registry),
            reader,
            registry,
        )
        # The view pins the segment: nothing reclaimed yet.
        self.assertEqual(pool.reclaimed, 0)
        del out
        gc.collect()
        # Once the view is gone, the segment returns to the pool.
        self.assertEqual(pool.reclaimed, 1)

    def test_torch_restore_is_a_zero_copy_view(self) -> None:
        """A restored Torch tensor aliases the segment (no copy)."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        pool = self._pool(1 << 20, 2)
        writer, reader = pool.open_writer(), pool.open_reader()
        registry = _default_registry()
        out = _restore(
            _offload({"a": torch.arange(1000, dtype=torch.int64)}, writer, registry),
            reader,
            registry,
        )
        view = cast(dict[str, Any], out)["a"]
        # Mutating the segment is observed through the tensor -> zero-copy view.
        struct.pack_into("<q", pool._segment(0), 0, 999)
        self.assertEqual(int(view[0]), 999)
        del out, view
        gc.collect()

    def test_torch_segment_held_until_view_released(self) -> None:
        """A Torch view outliving the original tensor still pins the segment."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        pool = self._pool(1 << 20, 2)
        writer, reader = pool.open_writer(), pool.open_reader()
        registry = _default_registry()
        out = _restore(
            _offload({"a": torch.arange(1000, dtype=torch.int64)}, writer, registry),
            reader,
            registry,
        )
        view = cast(dict[str, Any], out)["a"][100:200]  # a view sharing storage
        del out
        gc.collect()
        self.assertEqual(pool.reclaimed, 0)
        del view
        gc.collect()
        self.assertEqual(pool.reclaimed, 1)

    def test_segment_held_until_shallow_copy_released(self) -> None:
        """A view/shallow copy outliving the original array still pins the segment."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not available")
        pool = self._pool(1 << 20, 2)
        writer, reader = pool.open_writer(), pool.open_reader()
        registry = _default_registry()
        out = _restore(
            _offload({"a": np.arange(1000, dtype=np.int64)}, writer, registry),
            reader,
            registry,
        )
        # A slice shares the segment's memory but is a different object whose
        # base chain skips the array we first received.
        view = cast(dict[str, Any], out)["a"][100:200]
        del out
        gc.collect()
        # The lifetime is tracked on the underlying memoryview, not the array
        # object, so the segment is still held while the slice is alive.
        self.assertEqual(pool.reclaimed, 0)
        del view
        gc.collect()
        self.assertEqual(pool.reclaimed, 1)

    def test_bytes_like_restore_is_a_zero_copy_view(self) -> None:
        """On the pool, bytes/bytearray/memoryview all restore as a memoryview
        aliasing the segment, held until the view is released."""
        for label, src in (
            ("bytes", b"\xab" * 100_000),
            ("bytearray", bytearray(b"\xcd" * 100_000)),
            ("memoryview", memoryview(b"\xef" * 100_000)),
        ):
            with self.subTest(input=label):
                pool = self._pool(1 << 20, 2)
                writer, reader = pool.open_writer(), pool.open_reader()
                registry = _default_registry()
                out = _restore(_offload({"b": src}, writer, registry), reader, registry)
                view = cast(dict[str, Any], out)["b"]
                # Every input type comes back as a memoryview (the zero-copy path).
                self.assertIsInstance(view, memoryview)
                self.assertEqual(bytes(view), bytes(src))
                # The view aliases shared memory: a write to the segment shows
                # through, proving no copy was made.
                pool._segment(0)[0:1] = b"\x00"
                self.assertEqual(view[0], 0)
                # The segment stays reserved until the view is released.
                self.assertEqual(pool.reclaimed, 0)
                del out, view
                gc.collect()
                self.assertEqual(pool.reclaimed, 1)


class IterateInSubprocessSegmentPoolTest(unittest.TestCase):
    def test_round_trip_through_segment_pool(self) -> None:
        """Items round-trip end-to-end through a segment-pool arena.

        The pool restores offloaded bytes as zero-copy memoryviews, so each item
        is materialized (releasing its view) before the next is pulled, keeping the
        small pool from being exhausted by held views.
        """
        pool = SharedMemorySegmentPool(segment_size=1 << 18, count=4)
        got = []
        for item in iterate_in_subprocess(_make_items, arena=pool, buffer_size=2):
            got.append(
                {
                    k: bytes(v) if isinstance(v, memoryview) else v
                    for k, v in item.items()
                }
            )
            del item  # release the zero-copy view so its segment can be reclaimed
        self.assertEqual(got, _make_items())
