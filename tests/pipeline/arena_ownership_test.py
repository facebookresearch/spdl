# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gc
import struct
import unittest
import weakref
from typing import cast

import numpy as np
import torch
from spdl.io.lib import _buffer
from spdl.pipeline._arena._offload import _offload, _restore
from spdl.pipeline._arena._ownership import _export_buffer
from spdl.pipeline._arena._pool import SharedMemorySegmentPool
from spdl.pipeline._arena._registry import (
    _BytesHandler,
    _default_registry,
    _OffloadRegistry,
    _TorchHandler,
)


class ArenaOwnershipTest(unittest.TestCase):
    def _pool(self, count: int = 1) -> SharedMemorySegmentPool:
        pool = SharedMemorySegmentPool(8192, count, acquire_timeout=0)
        self.addCleanup(pool.unlink)
        self.addCleanup(pool.close)
        return pool

    def test_native_tensor_aliases_block_forced_reuse(self) -> None:
        """All detached, reshaped, transposed and storage aliases prevent overwrite."""
        aliases = {
            "detach": lambda t: t.detach(),
            "reshape": lambda t: t.detach().reshape(32, 32),
            "transpose": lambda t: t.detach().reshape(32, 32).T,
            "storage": lambda t: t.untyped_storage(),
            "set": lambda t: torch.empty(0, dtype=t.dtype).set_(t.untyped_storage()),
            "numpy": lambda t: t.numpy()[::2],
            "memoryview": lambda t: memoryview(t.numpy()).cast("B")[8:],
        }
        for name, make_alias in aliases.items():
            with self.subTest(alias=name):
                pool = self._pool()
                writer, reader = pool.open_writer(), pool.open_reader()
                registry = _default_registry()
                original = _restore(
                    _offload(torch.arange(1024), writer, registry), reader, registry
                )
                alias = make_alias(original)
                del original
                gc.collect()
                self.assertEqual(pool._reclaimed, 0)
                with self.assertRaises(BufferError):
                    writer.begin_unit()
                del alias
                gc.collect()
                self.assertEqual(pool._reclaimed, 1)
                replacement = _restore(
                    _offload(torch.full((1024,), 91), writer, registry),
                    reader,
                    registry,
                )
                assert isinstance(replacement, torch.Tensor)
                self.assertEqual(cast(torch.Tensor, replacement).tolist(), [91] * 1024)
                del replacement
                gc.collect()
                self.assertEqual(pool._reclaimed, 2)

    def test_buffer_and_numpy_aliases_hold_exporter(self) -> None:
        """Slice/cast and cross-library aliases retain the actual exporter."""
        cases = {
            "slice": lambda v: v[::2],
            "cast": lambda v: v.cast("q"),
            "numpy": lambda v: np.frombuffer(v, dtype=np.int64)[::2],
            "torch_numpy": lambda v: torch.from_numpy(
                np.frombuffer(v, dtype=np.int64)
            ).detach(),
        }
        for name, make_alias in cases.items():
            with self.subTest(alias=name):
                pool = self._pool()
                writer, reader = pool.open_writer(), pool.open_reader()
                registry = _default_registry()
                value = _restore(
                    _offload(b"a" * 8192, writer, registry), reader, registry
                )
                alias = make_alias(value)
                del value
                gc.collect()
                self.assertEqual(pool._reclaimed, 0)
                with self.assertRaises(BufferError):
                    writer.begin_unit()
                del alias
                gc.collect()
                self.assertEqual(pool._reclaimed, 1)

    def test_close_defers_unmap_until_last_storage_release(self) -> None:
        """Close/unlink keeps native storage valid through a final safe retirement retry."""
        pool = self._pool()
        registry = _default_registry()
        value = _restore(
            _offload(torch.arange(1024), pool.open_writer(), registry),
            pool.open_reader(),
            registry,
        )
        assert isinstance(value, torch.Tensor)
        storage = cast(torch.Tensor, value).untyped_storage()
        del value
        gc.collect()
        pool.close()
        with self.assertRaises(BufferError):
            pool.unlink()
        self.assertIsNotNone(pool._segs)
        alias = torch.empty(0, dtype=torch.int64).set_(storage)
        self.assertEqual(alias.tolist(), list(range(1024)))
        del storage, alias
        gc.collect()
        pool.close()
        self.assertIsNone(pool._segs)

    def test_restore_failure_retains_escaped_buffer(self) -> None:
        """A failing handler cannot reclaim a buffer it already exposed."""
        escaped = []

        class FailingHandler(_BytesHandler):
            def from_buffer(self, buf, meta, zero_copy):
                escaped.append(buf[1:])
                raise ValueError("restore failure")

        registry = _OffloadRegistry([FailingHandler()])
        pool = self._pool()
        writer, reader = pool.open_writer(), pool.open_reader()
        with self.assertRaisesRegex(ValueError, "restore failure"):
            _restore(_offload(b"z" * 8192, writer, registry), reader, registry)
        gc.collect()
        self.assertEqual(pool._reclaimed, 0)
        self.assertEqual(bytes(escaped[0]), b"z" * 8191)
        escaped.clear()
        gc.collect()
        self.assertEqual(pool._reclaimed, 1)

    def test_release_order_and_multiple_exports(self) -> None:
        """Out-of-order unit release advances only the contiguous watermark."""
        pool = self._pool(2)
        writer, reader = pool.open_writer(), pool.open_reader()
        registry = _default_registry()
        first = _restore(
            _offload([b"a" * 4096, b"b" * 4096], writer, registry), reader, registry
        )
        second = _restore(_offload(b"c" * 8192, writer, registry), reader, registry)
        assert isinstance(first, list)
        held = cast(list[memoryview], first)[0][1:].cast("B")
        del first, second
        gc.collect()
        self.assertEqual(pool._reclaimed, 0)
        self.assertEqual(bytes(held), b"a" * 4095)
        del held
        gc.collect()
        self.assertEqual(pool._reclaimed, 2)

    def test_explicit_release_keeps_native_exporter_consumer_alive(self) -> None:
        """The safe raw-exporter Torch route survives release of the original view."""
        pool = self._pool()
        writer, reader = pool.open_writer(), pool.open_reader()
        registry = _default_registry()
        view = _restore(_offload(b"a" * 8192, writer, registry), reader, registry)
        assert isinstance(view, memoryview)
        tensor = torch.frombuffer(view.obj, dtype=torch.uint8).detach()
        array = np.frombuffer(view, dtype=np.uint8)[::2]
        view.release()
        del view
        gc.collect()
        self.assertEqual(pool._reclaimed, 0)
        self.assertTrue(torch.all(tensor == 97).item())
        np.testing.assert_array_equal(array, np.full(4096, 97, dtype=np.uint8))
        del tensor, array
        gc.collect()
        self.assertEqual(pool._reclaimed, 1)

    def test_failed_export_does_not_take_callback_ownership(self) -> None:
        """Native construction failures do not invoke the release callback."""
        released = []
        with self.assertRaises(BufferError):
            _export_buffer(memoryview(b"readonly"), lambda: released.append(True))
        gc.collect()
        self.assertEqual(released, [])

    def test_owned_allocator_export_retains_original_native_allocation(self) -> None:
        """Native allocation survives detached aliases of restored views."""
        source = torch.arange(1024, dtype=torch.int64)
        reference = weakref.ref(source)
        view = _buffer._export_owned(source.data_ptr(), 8192, source)
        restored = torch.frombuffer(view.obj, dtype=torch.int64)
        alias = restored.detach()
        del source, view, restored
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertEqual(alias.tolist(), list(range(1024)))
        del alias
        gc.collect()
        self.assertIsNone(reference())

    def test_sliced_native_export_keeps_correct_offset_and_private_wrapper(
        self,
    ) -> None:
        """Sliced views keep offsets after explicit original release."""
        released = []
        view = _export_buffer(
            memoryview(bytearray(struct.pack("8q", *range(8)))),
            lambda: released.append(True),
        )
        sliced = view[8:32]
        restored, _ = _TorchHandler().from_buffer(sliced, ("torch.int64", (3,)), False)
        sliced.release()
        del sliced, view
        gc.collect()
        self.assertEqual(cast(torch.Tensor, restored).tolist(), [1, 2, 3])
        self.assertEqual(released, [])
        del restored
        gc.collect()
        self.assertEqual(released, [True])

    def test_empty_export_notifies_once(self) -> None:
        """An empty export notifies once, after its last derived view is released."""
        released = []
        view = _export_buffer(memoryview(bytearray()), lambda: released.append(True))
        self.assertEqual(view.nbytes, 0)
        alias = view[:]
        del view
        gc.collect()
        self.assertEqual(released, [])
        del alias
        gc.collect()
        self.assertEqual(released, [True])
