# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
import unittest
from typing import Any, cast

from spdl.pipeline._arena._ownership import _export_buffer
from spdl.pipeline._arena._pool import SharedMemorySegmentPool


class ArenaNativeOwnershipTest(unittest.TestCase):
    def test_optional_io_exporter_does_not_load_media_or_torch(self) -> None:
        """Real IO initialization may load NumPy, but not CUDA/media or Torch."""
        script = textwrap.dedent("""
            import importlib.abc, sys
            class RejectHeavy(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == 'spdl.pipeline.fb':
                        raise ModuleNotFoundError(
                            'Meta logging is not part of the OSS fixture', name=fullname
                        )
                    if (
                        fullname.split('.')[0] in ('torch', 'numba')
                        or (fullname.startswith('spdl.io.lib.') and any(
                            s in fullname for s in ('_spdl', '_libspdl', '_archive', '_wav')
                        ))
                    ):
                        raise AssertionError('unexpected heavy import: ' + fullname)
            sys.meta_path.insert(0, RejectHeavy())
            from spdl.pipeline._arena._ownership import _export_buffer
            released = []
            view = _export_buffer(memoryview(bytearray(b'safe')), lambda: released.append(True))
            alias = view[::2]
            view.release(); del view
            assert bytes(alias) == b'sf' and not released
            alias.release(); del alias
            assert released == [True]
            from spdl.io.lib import _buffer
            import ctypes
            class Owner:
                def __init__(self):
                    self.storage = bytearray(b'owned')
                def __del__(self):
                    released.append('owner')
            owner = Owner()
            pointer = ctypes.addressof(ctypes.c_char.from_buffer(owner.storage))
            view = _buffer._export_owned(pointer, 5, owner)
            alias = view[1:]
            del owner
            view.release(); del view
            assert bytes(alias) == b'wned' and released == [True]
            alias.release(); del alias
            assert released == [True, 'owner']
            assert 'spdl.io.lib._buffer' in sys.modules
            assert 'torch' not in sys.modules
            assert not any(
                n.startswith('spdl.io.lib.') and any(
                    s in n for s in ('_spdl', '_libspdl', '_archive', '_wav')
                ) for n in sys.modules
            )
            print('OPTIONAL_IO_BUFFER_PASSED')
        """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OPTIONAL_IO_BUFFER_PASSED", result.stdout)

    def test_memoryview_aliases_hold_reusable_storage_without_dependencies(
        self,
    ) -> None:
        """Slices/casts hold optional IO storage without loading Torch or CUDA."""
        pool = SharedMemorySegmentPool(64, 1, acquire_timeout=0)
        self.addCleanup(pool.unlink)
        self.addCleanup(pool.close)
        writer, reader = pool.open_writer(), pool.open_reader()
        writer.begin_unit()
        offset, size = writer.write_binary(b"a" * 64)
        span = writer.commit_unit()
        view = reader.read_binary(offset, size)
        alias = view[::2]
        reader.end_unit(span, [])
        view.release()
        del view
        gc.collect()
        self.assertEqual(pool._reclaimed, 0)
        self.assertEqual(bytes(alias), b"a" * 32)
        with self.assertRaises(BufferError):
            writer.begin_unit()
        del alias
        gc.collect()
        self.assertEqual(pool._reclaimed, 1)
        self.assertNotIn("torch", sys.modules)
        self.assertIn("spdl.io.lib._buffer", sys.modules)

    def test_close_rejects_new_exports_and_preserves_retained_alias(self) -> None:
        """Closed pools cannot reopen mappings while old native exports remain valid."""
        pool = SharedMemorySegmentPool(64, 1, acquire_timeout=0)
        self.addCleanup(pool.unlink)
        writer, reader = pool.open_writer(), pool.open_reader()
        writer.begin_unit()
        offset, size = writer.write_binary(b"a" * 64)
        span = writer.commit_unit()
        view = reader.read_binary(offset, size)
        reader.end_unit(span, [])
        pool.close()
        pool.unlink()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            reader.read_binary(0, 1)
        self.assertEqual(bytes(view), b"a" * 64)
        del view
        gc.collect()
        self.assertIsNone(pool._segs)
        self.assertIsNone(pool._ctrl)

    def test_native_exporter_finalizes_after_resurrection(self) -> None:
        """Cyclic resurrection cannot release a still-usable exporter."""
        released = []
        resurrected = []
        view = _export_buffer(
            memoryview(bytearray(b"safe")), lambda: released.append(True)
        )
        with self.assertRaises(AttributeError):
            cast(Any, view.obj).__setattr__("owner", None)

        class Holder:
            view: "memoryview[bytes]"
            cycle: Holder | None

            def __init__(self, view: memoryview[bytes]) -> None:
                self.view = view
                self.cycle: Holder | None = self

            def __del__(self) -> None:
                resurrected.append(self)

        holder = Holder(view)
        del view, holder
        gc.collect()
        self.assertEqual(released, [])
        self.assertEqual(bytes(resurrected[0].view), b"safe")
        resurrected[0].cycle = None
        resurrected.clear()
        gc.collect()
        self.assertEqual(released, [True])

    def test_failed_and_empty_exports_have_transactional_callbacks(self) -> None:
        """Failure leaves ownership with caller; empty success releases exactly once."""
        released = []
        with self.assertRaises(BufferError):
            _export_buffer(memoryview(b"readonly"), lambda: released.append(True))
        self.assertEqual(released, [])
        view = _export_buffer(memoryview(bytearray()), lambda: released.append(True))
        self.assertEqual(view.nbytes, 0)
        del view
        gc.collect()
        self.assertEqual(released, [True])

    def _post_factory_failure(self, first_normal: bool, escaped: bool) -> None:
        from unittest.mock import patch

        from spdl.pipeline._arena import _ownership

        pool = SharedMemorySegmentPool(256, 1, acquire_timeout=0)
        self.addCleanup(pool.unlink)
        self.addCleanup(pool.close)
        writer, reader = pool.open_writer(), pool.open_reader()
        writer.begin_unit()
        offset, size = writer.write_binary(b"a" * 32)
        span = writer.commit_unit()
        first = [reader.read_binary(offset, size)] if first_normal else []
        root = MemoryError("after actual native exporter construction")
        captured = []
        self.addCleanup(setattr, root, "__traceback__", None)
        self.addCleanup(captured.clear)
        self.addCleanup(first.clear)

        def fail_after_native(_type, value):
            self.assertEqual(type(value.obj).__module__, "spdl.io.lib._buffer")
            self.assertEqual(bytes(value), b"a" * 32)
            if escaped:
                captured.append(value[1:])
            raise root

        with patch.object(_ownership, "cast", new=fail_after_native):
            try:
                reader.read_binary(offset, size)
            except MemoryError as error:
                self.assertIs(error, root)
            else:
                self.fail("native post-processing did not fail")
        name = pool._seg_names[0]
        self.assertEqual(pool._mapping_exports[name], 1 + int(first_normal))
        reader.end_unit(span, [])
        if first:
            self.assertEqual(bytes(first[0]), b"a" * 32)
            first[0].release()
            first.clear()
        gc.collect()
        self.assertEqual(pool._mapping_exports[name], 1)
        self.assertEqual(pool._reclaimed, 0)
        with self.assertRaises(BufferError):
            writer.begin_unit()
        self._release_failed_export(root, captured, pool, writer, name)

    def _release_failed_export(self, root, captured, pool, writer, name) -> None:
        root.__traceback__ = None
        gc.collect()
        if captured:
            self.assertEqual(bytes(captured[0]), b"a" * 31)
            self.assertEqual(pool._mapping_exports[name], 1)
            with self.assertRaises(BufferError):
                writer.begin_unit()
            captured.clear()
            gc.collect()
        self.assertEqual(pool._mapping_exports[name], 0)
        self.assertEqual(pool._reclaimed, 1)
        writer.begin_unit()
        writer.write_binary(b"b" * 32)
        writer.abort_unit()

    def test_failed_return_traceback_alone_keeps_native_export(self) -> None:
        """The exception traceback is an actual consumer of the adopted export."""
        self._post_factory_failure(first_normal=False, escaped=False)

    def test_failed_return_outlives_other_successful_export(self) -> None:
        """Releasing another binary does not retire traceback-owned storage."""
        self._post_factory_failure(first_normal=True, escaped=False)

    def test_failed_return_escaped_alias_survives_exception_cleanup(self) -> None:
        """An escaped alias remains live after clearing the original exception."""
        self._post_factory_failure(first_normal=True, escaped=True)

    def test_pinned_failed_return_delays_unregister_until_traceback_release(
        self,
    ) -> None:
        """An error-owned export remains registered after another alias and close."""
        from unittest.mock import Mock, patch

        from spdl.pipeline._arena import _ownership

        registration = Mock()
        with (
            patch(
                "spdl.pipeline._arena._pool._HostRegistration",
                return_value=registration,
            ),
            patch("spdl.pipeline._arena._pool._local_device", return_value="cuda:0"),
            patch("spdl.pipeline._arena._pool._check_host_budget"),
        ):
            pool = SharedMemorySegmentPool(256, 1, acquire_timeout=0, pin_memory=True)
            self.addCleanup(pool.unlink)
            self.addCleanup(pool.close)
            writer, reader = pool.open_writer(), pool.open_reader()
            writer.begin_unit()
            offset, size = writer.write_binary(b"safe" * 8)
            span = writer.commit_unit()
            first = reader.read_binary(offset, size)
            root = MemoryError("after pinned native handoff")
            self.addCleanup(setattr, root, "__traceback__", None)

            def fail_after_native(_type, value):
                self.assertEqual(bytes(value), b"safe" * 8)
                raise root

            with patch.object(_ownership, "cast", fail_after_native):
                try:
                    reader.read_binary(offset, size)
                except MemoryError as error:
                    self.assertIs(error, root)
                else:
                    self.fail("post-native failure was not injected")
            reader.end_unit(span, [])
            first.release()
            del first
            pool.close()
            gc.collect()
            name = pool._seg_names[0]
            self.assertEqual(pool._mapping_exports[name], 1)
            registration.close.assert_not_called()
            self.assertIsNotNone(pool._segs)
            root.__traceback__ = None
            gc.collect()
            self.assertEqual(pool._mapping_exports[name], 0)
            registration.close.assert_called_once()
            # Traceback clearing can drop the returned native view before a
            # temporary source slice in another frame. Retry close after all
            # those private frames are gone; unregister must not run twice.
            pool.close()
            registration.close.assert_called_once()
            self.assertIsNone(pool._segs)

    def test_ticket_rejected_reuse_does_not_release_first_export(self) -> None:
        """Failed native adoption cannot retire another export's capability."""
        from spdl.io.lib import _buffer

        released = []
        ticket = _buffer._release_ticket(lambda: released.append(True))
        ticket._activate()
        view = _buffer._export(memoryview(bytearray(b"safe")), ticket)
        alias = view[1:]
        view.release()
        del view
        ticket._rollback()
        self.assertEqual(released, [])
        with self.assertRaisesRegex(RuntimeError, "setup-owned"):
            _buffer._export(memoryview(bytearray(b"other")), ticket)
        with self.assertRaisesRegex(RuntimeError, "native storage"):
            ticket._cancel()
        self.assertEqual(bytes(alias), b"afe")
        self.assertEqual(released, [])
        alias.release()
        del alias
        gc.collect()
        self.assertEqual(released, [True])
        ticket._rollback()
        self.assertEqual(released, [True])

    def test_ticket_setup_failure_and_cancel_are_transactional(self) -> None:
        """A failed pre-adoption export rolls back once; canceled setup calls none."""
        from spdl.io.lib import _buffer

        released = []
        ticket = _buffer._release_ticket(lambda: released.append(True))
        ticket._activate()
        with self.assertRaises(BufferError):
            _buffer._export(memoryview(b"readonly"), ticket)
        self.assertEqual(released, [])
        ticket._rollback()
        ticket._rollback()
        self.assertEqual(released, [True])
        canceled = _buffer._release_ticket(lambda: released.append(False))
        canceled._activate()
        canceled._cancel()
        canceled._rollback()
        self.assertEqual(released, [True])
        with self.assertRaises(AttributeError):
            cast(Any, ticket).__setattr__("release", None)

    def test_ticket_native_owner_survives_cyclic_resurrection(self) -> None:
        """Native-owned tickets retain storage through a resurrected view holder."""
        from spdl.io.lib import _buffer

        released = []
        resurrected = []
        ticket = _buffer._release_ticket(lambda: released.append(True))
        ticket._activate()
        view = _buffer._export(memoryview(bytearray(b"safe")), ticket)

        class Holder:
            def __init__(self, value):
                self.view = value
                self.cycle = self

            def __del__(self):
                resurrected.append(self)

        holder = Holder(view)
        del view, holder
        gc.collect()
        ticket._rollback()
        self.assertEqual(released, [])
        self.assertEqual(bytes(resurrected[0].view), b"safe")
        resurrected[0].cycle = None
        resurrected.clear()
        gc.collect()
        self.assertEqual(released, [True])

    def test_failed_ticket_activation_restores_acquired_counts(self) -> None:
        """Failure after native setup activation cancels only restored setup holds."""
        from unittest.mock import patch

        from spdl.pipeline._arena._ownership import _ExportGuard

        pool = SharedMemorySegmentPool(256, 1, acquire_timeout=0)
        self.addCleanup(pool.unlink)
        self.addCleanup(pool.close)
        writer, reader = pool.open_writer(), pool.open_reader()
        writer.begin_unit()
        offset, size = writer.write_binary(b"a" * 32)
        span = writer.commit_unit()
        root = MemoryError("after setup ticket activation")
        activate = _ExportGuard.activate

        def fail_after_activation(guard):
            activate(guard)
            raise root

        with patch.object(_ExportGuard, "activate", new=fail_after_activation):
            with self.assertRaises(MemoryError) as raised:
                reader.read_binary(offset, size)
        self.assertIs(raised.exception, root)
        self.assertEqual(pool._mapping_exports.get(pool._seg_names[0], 0), 0)
        reader.end_unit(span, [])
        self.assertEqual(pool._reclaimed, 1)
        writer.begin_unit()
        writer.abort_unit()

    def test_ticket_empty_export_and_concurrent_rollback(self) -> None:
        """Setup releases once; concurrent rollback cannot retire a live empty export."""
        import threading

        from spdl.io.lib import _buffer

        released = []
        setup = _buffer._release_ticket(lambda: released.append("setup"))
        setup._activate()
        native = _buffer._release_ticket(lambda: released.append("native"))
        native._activate()
        view = _buffer._export(memoryview(bytearray()), native)
        alias = view[:]
        view.release()
        threads = [
            threading.Thread(target=ticket._rollback)
            for ticket in (setup, native)
            for _ in range(16)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
            self.assertFalse(thread.is_alive())
        self.assertEqual(released, ["setup"])
        with self.assertRaises(TypeError):
            cast(Any, type(native)).__setattr__(type(native), "_rollback", None)
        alias.release()
        self.assertEqual(released, ["setup", "native"])

    def test_native_retirement_acquires_pool_before_reader(self) -> None:
        """A real native callback uses one pool-to-reader order across threads."""
        import threading
        from unittest.mock import patch

        pool = SharedMemorySegmentPool(256, 1, acquire_timeout=0)
        self.addCleanup(pool.unlink)
        self.addCleanup(pool.close)
        writer, reader = pool.open_writer(), pool.open_reader()
        writer.begin_unit()
        offset, size = writer.write_binary(b"a" * 32)
        span = writer.commit_unit()
        view = reader.read_binary(offset, size)
        reader.end_unit(span, [])
        reader_lock = reader._lock
        order = []

        class ObservedLock:
            def __enter__(self):
                order.append(pool._close_lock._is_owned())
                reader_lock.acquire()

            def __exit__(self, *args):
                reader_lock.release()

        with patch.object(reader, "_lock", ObservedLock()):
            worker = threading.Thread(target=view.release)
            worker.start()
            worker.join(10)
            self.assertFalse(worker.is_alive())
        self.assertEqual(order, [True])
        self.assertEqual(pool._reclaimed, 1)

    def test_gc_keeps_bound_partial_callback_callable_until_export_destruction(
        self,
    ) -> None:
        """Collecting a cyclic view does not clear its still-needed callback graph."""
        from functools import partial

        released = []

        class Receiver:
            def notify(self, tag):
                released.append(tag)

        receiver = Receiver()
        view = _export_buffer(
            memoryview(bytearray(b"safe")), partial(receiver.notify, "released")
        )
        cycle: list[object] = [view]
        cycle.append(cycle)
        del receiver, view, cycle
        gc.collect()
        self.assertEqual(released, ["released"])

    def test_cyclic_native_callback_reenters_held_retirement_locks(self) -> None:
        """Cyclic GC may retire storage while the same thread owns both locks."""
        pool = SharedMemorySegmentPool(256, 1, acquire_timeout=0)
        self.addCleanup(pool.unlink)
        self.addCleanup(pool.close)
        writer, reader = pool.open_writer(), pool.open_reader()
        writer.begin_unit()
        offset, size = writer.write_binary(b"a" * 32)
        span = writer.commit_unit()
        view = reader.read_binary(offset, size)
        reader.end_unit(span, [])
        enabled = gc.isenabled()
        gc.disable()
        try:
            cycle: list[object] = [view]
            cycle.append(cycle)
            del view, cycle
            with pool._close_lock, reader._lock:
                reentrant = reader._lock.acquire(blocking=False)
                self.assertTrue(reentrant)
                reader._lock.release()
                self.assertEqual(pool._mapping_exports[pool._seg_names[0]], 1)
                gc.collect()
                self.assertEqual(pool._mapping_exports[pool._seg_names[0]], 0)
                self.assertEqual(pool._reclaimed, 1)
        finally:
            if enabled:
                gc.enable()
            gc.collect()

    def test_one_counter_failure_restores_only_setup_transaction(self) -> None:
        """A fault after the first accounting write cancels the unused ticket."""
        pool = SharedMemorySegmentPool(256, 1, acquire_timeout=0)
        self.addCleanup(pool.unlink)
        self.addCleanup(pool.close)
        writer, reader = pool.open_writer(), pool.open_reader()
        writer.begin_unit()
        offset, size = writer.write_binary(b"a" * 32)
        span = writer.commit_unit()
        root = MemoryError("after mapping counter commit")

        class FailAfterSet(dict):
            armed = True

            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                if self.armed:
                    self.armed = False
                    raise root

        pool._mapping_exports = FailAfterSet()
        with self.assertRaises(MemoryError) as raised:
            reader.read_binary(offset, size)
        self.assertIs(raised.exception, root)
        self.assertEqual(pool._mapping_exports[pool._seg_names[0]], 0)
        reader.end_unit(span, [])
        self.assertEqual(pool._reclaimed, 1)
        writer.begin_unit()
        writer.abort_unit()
