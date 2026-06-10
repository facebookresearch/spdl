# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Unit tests for the per-type offload handlers in :py:mod:`_registry`.

These exercise each handler's ``matches`` gating and its ``get_buffer`` /
``from_buffer`` contract directly, without going through a shared-memory backend.
``from_buffer`` is fed a plain ``memoryview`` standing in for the arena buffer: a
writable ``bytearray``-backed view models both the ring's private per-unit copy
(``zero_copy=False``) and the pool's live shared-memory segment
(``zero_copy=True``). End-to-end round-trips through a real backend live in the
ring / pool / packets test files.
"""

import unittest
from typing import Any, cast

import spdl.io
from spdl.pipeline._arena._registry import (
    _BytesHandler,
    _NumpyHandler,
    _PacketsHandler,
    _TorchHandler,
)

from ..fixture import FFMPEG_CLI, get_sample

_VIDEO_CMD = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 25 sample.mp4"


def _demux_video() -> Any:
    """Generate a synthetic clip and demux it into a VideoPackets object."""
    # Keep the SrcInfo alive until demux opens the file: dropping it would run
    # its TemporaryDirectory finalizer and delete sample.mp4 first.
    sample = get_sample(_VIDEO_CMD)
    return spdl.io.demux_video(sample.path)


def _copy_view(buf: "memoryview[bytes] | bytes | bytearray") -> "memoryview[bytes]":
    """A writable, private copy of ``buf`` as a memoryview (models arena bytes)."""
    return memoryview(bytearray(buf))


class BytesHandlerTest(unittest.TestCase):
    def test_matches_accepts_large_bytes_like(self) -> None:
        """bytes, bytearray and memoryview at/above the threshold are claimed."""
        handler = _BytesHandler()
        blob = b"x" * 100
        self.assertTrue(handler.matches(blob, 100))
        self.assertTrue(handler.matches(bytearray(blob), 100))
        self.assertTrue(handler.matches(memoryview(blob), 100))

    def test_matches_rejects_below_threshold(self) -> None:
        """A bytes-like smaller than the threshold is left inline."""
        self.assertFalse(_BytesHandler().matches(b"x" * 99, 100))

    def test_matches_rejects_non_bytes_like(self) -> None:
        """Objects that are not bytes-like are not claimed by this handler."""
        handler = _BytesHandler()
        self.assertFalse(handler.matches("a string", 0))
        self.assertFalse(handler.matches([1, 2, 3], 0))
        self.assertFalse(handler.matches(12345, 0))

    def test_copy_restore_reconstructs_original_type(self) -> None:
        """zero_copy=False (ring) rebuilds each input as its own type, no anchor."""
        handler = _BytesHandler()
        cases: list[tuple[Any, type[Any]]] = [
            (b"a" * 50, bytes),
            (bytearray(b"b" * 50), bytearray),
            (memoryview(b"c" * 50), memoryview),
        ]
        for src, expected_type in cases:
            with self.subTest(input=expected_type.__name__):
                buf, meta = handler.get_buffer(src)
                restored, anchor = handler.from_buffer(_copy_view(buf), meta, False)
                self.assertIsInstance(restored, expected_type)
                self.assertEqual(bytes(cast(Any, restored)), bytes(src))
                self.assertIsNone(anchor)

    def test_zero_copy_restore_is_a_self_anchoring_memoryview(self) -> None:
        """zero_copy=True (pool) returns a memoryview that is its own anchor."""
        buf, meta = _BytesHandler().get_buffer(b"z" * 64)
        restored, anchor = _BytesHandler().from_buffer(_copy_view(buf), meta, True)
        self.assertIsInstance(restored, memoryview)
        self.assertIs(restored, anchor)

    def test_noncontiguous_memoryview_is_flattened(self) -> None:
        """get_buffer copies a non-contiguous memoryview into a flat buffer."""
        src = memoryview(bytes(range(256)) * 4)[::2]  # strided -> non-contiguous
        self.assertFalse(src.contiguous)
        buf, _ = _BytesHandler().get_buffer(src)
        self.assertTrue(memoryview(buf).contiguous)
        self.assertEqual(bytes(buf), bytes(src))


class NumpyHandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import numpy as np

            self.np = np
        except ImportError:
            self.skipTest("numpy not available")

    def test_matches_rejects_non_numpy(self) -> None:
        """A foreign type (even one named 'ndarray') is not claimed."""
        handler = _NumpyHandler()

        class ndarray:  # noqa: N801 — deliberately mimics numpy.ndarray
            nbytes: int = 1 << 30

        self.assertFalse(handler.matches(ndarray(), 0))
        self.assertFalse(handler.matches(b"bytes", 0))

    def test_matches_respects_threshold(self) -> None:
        """An ndarray is claimed only when its nbytes reaches the threshold."""
        arr = self.np.zeros(100, dtype=self.np.float64)  # 800 bytes
        handler = _NumpyHandler()
        self.assertTrue(handler.matches(arr, 800))
        self.assertFalse(handler.matches(arr, 801))

    def test_round_trip_copy_and_zero_copy(self) -> None:
        """get_buffer/from_buffer rebuild an equal array; only the zero-copy path
        returns a lifetime anchor."""
        np = self.np
        arr = np.arange(256, dtype=np.float64).reshape(16, 16)
        buf, meta = _NumpyHandler().get_buffer(arr)

        copied, anchor = _NumpyHandler().from_buffer(_copy_view(buf), meta, False)
        copied = cast(Any, copied)
        self.assertTrue(np.array_equal(copied, arr))
        self.assertEqual(copied.dtype, arr.dtype)
        self.assertEqual(copied.shape, arr.shape)
        self.assertIsNone(anchor)

        viewed, view_anchor = _NumpyHandler().from_buffer(_copy_view(buf), meta, True)
        self.assertTrue(np.array_equal(cast(Any, viewed), arr))
        self.assertIsNotNone(view_anchor)


class TorchHandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch

            self.torch = torch
        except ImportError:
            self.skipTest("torch not available")

    def test_matches_rejects_non_torch(self) -> None:
        """A foreign type (even one named 'Tensor') is not claimed."""
        handler = _TorchHandler()

        class Tensor:  # deliberately mimics torch.Tensor
            is_cpu: bool = True

        self.assertFalse(handler.matches(Tensor(), 0))
        self.assertFalse(handler.matches(b"bytes", 0))

    def test_matches_respects_threshold(self) -> None:
        """A CPU tensor is claimed only when its byte size reaches the threshold."""
        t = self.torch.zeros(100, dtype=self.torch.int64)  # 800 bytes
        handler = _TorchHandler()
        self.assertTrue(handler.matches(t, 800))
        self.assertFalse(handler.matches(t, 801))

    def test_round_trip_copy_and_zero_copy(self) -> None:
        """get_buffer/from_buffer rebuild an equal tensor; only the zero-copy path
        returns a lifetime anchor."""
        torch = self.torch
        t = torch.arange(256, dtype=torch.int64).reshape(16, 16)
        buf, meta = _TorchHandler().get_buffer(t)

        copied, anchor = _TorchHandler().from_buffer(_copy_view(buf), meta, False)
        self.assertTrue(torch.equal(cast(Any, copied), t))
        self.assertIsNone(anchor)

        viewed, view_anchor = _TorchHandler().from_buffer(_copy_view(buf), meta, True)
        self.assertTrue(torch.equal(cast(Any, viewed), t))
        self.assertIsNotNone(view_anchor)


class PacketsHandlerTest(unittest.TestCase):
    def test_foreign_type_sharing_the_name_is_rejected(self) -> None:
        """A non-Packets class that merely reuses the name is not matched, and its
        __getstate__ (which may not return bytes) is never called."""

        class AudioPackets:  # noqa: B903 — deliberately mimics the Packets name
            def __getstate__(self) -> dict[str, str]:
                raise AssertionError(
                    "__getstate__ must not be called on a foreign type"
                )

        handler = _PacketsHandler()
        # threshold=0 would match anything that passes the type gate.
        self.assertFalse(handler.matches(AudioPackets(), 0))
        self.assertIsNone(handler._cache)

    def test_sub_threshold_packets_are_not_cached(self) -> None:
        """A genuine Packets below the threshold is left inline and leaves no
        serialized bytes pinned in the handler cache."""
        handler = _PacketsHandler()
        packets = _demux_video()
        above_size = len(packets.__getstate__()) + 1
        self.assertFalse(handler.matches(packets, above_size))
        self.assertIsNone(handler._cache)

    def test_matched_packets_cache_is_consumed_by_get_buffer(self) -> None:
        """An above-threshold Packets matches, caches its bytes for the following
        get_buffer, and the cache is cleared once get_buffer consumes it."""
        handler = _PacketsHandler()
        packets = _demux_video()
        self.assertTrue(handler.matches(packets, 0))
        self.assertIsNotNone(handler._cache)
        buf, meta = handler.get_buffer(packets)
        self.assertEqual(bytes(buf), packets.__getstate__())
        self.assertIs(meta[0], type(packets))
        self.assertIsNone(handler._cache)

    def test_copy_restore_round_trips(self) -> None:
        """zero_copy=False rebuilds an owning Packets equal to the original."""
        handler = _PacketsHandler()
        packets = _demux_video()
        buf, meta = handler.get_buffer(packets)
        restored, anchor = handler.from_buffer(_copy_view(buf), meta, False)
        self.assertIs(type(restored), type(packets))
        self.assertEqual(cast(Any, restored).__getstate__(), packets.__getstate__())
        self.assertIsNone(anchor)


if __name__ == "__main__":
    unittest.main()
