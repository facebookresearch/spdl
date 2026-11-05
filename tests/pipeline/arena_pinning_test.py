# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import ctypes
import gc
import os
import tempfile
import unittest
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from spdl.pipeline._arena import _pinning
from spdl.pipeline._arena._pool import SharedMemorySegmentPool
from spdl.pipeline._arena._ring import SharedMemoryRingBuffer


def _retry_cleanup(reference, registrations) -> None:
    for registration in registrations:
        registration.close.side_effect = None
    pool = reference()
    if pool is not None:
        pool.close()
        pool.unlink()


class ArenaPinningTest(unittest.TestCase):
    def _pool(self, registrations: list[Mock], *, cleanup: bool = True):
        factory = self.enterContext(
            patch(
                "spdl.pipeline._arena._pool._HostRegistration",
                side_effect=registrations,
            )
        )
        self.enterContext(
            patch("spdl.pipeline._arena._pool._local_device", return_value="cuda:1")
        )
        self.enterContext(patch("spdl.pipeline._arena._pool._check_host_budget"))
        pool = SharedMemorySegmentPool(
            64, len(registrations), acquire_timeout=0, pin_memory=True
        )
        if cleanup:
            self.addCleanup(pool.unlink)
            self.addCleanup(pool.close)
        else:
            self.addCleanup(_retry_cleanup, weakref.ref(pool), registrations)
        return pool, factory

    def _publish(self, pool):
        writer, reader = pool.open_writer(), pool.open_reader()
        writer.begin_unit()
        off, size = writer.write_binary(b"a" * 64)
        span = writer.commit_unit()
        value = reader.read_binary(off, size)
        reader.end_unit(span, [])
        return value

    def test_pool_registration_is_consumer_local_and_reused(self) -> None:
        """Creation/writing never registers; the consumer registers each mapping once."""
        registrations = [Mock(), Mock()]
        pool, factory = self._pool(registrations)
        writer, reader = pool.open_writer(), pool.open_reader()
        self.assertEqual(factory.call_count, 0)
        for tag in range(8):
            writer.begin_unit()
            off, size = writer.write_binary(bytes([tag]) * 64)
            span = writer.commit_unit()
            value = reader.read_binary(off, size)
            reader.end_unit(span, [])
            self.assertEqual(bytes(value), bytes([tag]) * 64)
            del value
            gc.collect()
        self.assertEqual(factory.call_count, 2)
        for registration in registrations:
            registration.register.assert_called_once()
        self.assertNotIn("_registrations", pool.__getstate__())

    def test_pool_close_waits_for_exports_before_unregister(self) -> None:
        """Final derived export release unregisters before unmap, even after unlink."""
        registration = Mock()
        pool, _ = self._pool([registration])
        value = self._publish(pool)
        alias = value[1:]
        del value
        pool.close()
        pool.unlink()
        registration.close.assert_not_called()
        self.assertEqual(bytes(alias), b"a" * 63)
        del alias
        gc.collect()
        registration.close.assert_called_once()
        self.assertIsNone(pool._segs)

    def test_repeated_close_retains_exports_and_retries_failed_unregister(self) -> None:
        """Both retained-segment close branches keep mappings until safe cleanup."""
        registration = Mock()
        pool, _ = self._pool([registration])
        value = self._publish(pool)
        alias = value[1:]
        value.release()
        del value
        assert pool._segs is not None
        names = [segment.name for segment in pool._segs]
        for _ in range(3):
            pool.close()
            assert pool._segs is not None
            self.assertEqual([segment.name for segment in pool._segs], names)
            registration.close.assert_not_called()
            self.assertEqual(bytes(alias), b"a" * 63)
        registration.close.side_effect = RuntimeError("unregister retry root")
        with self.assertLogs("spdl.pipeline._arena._pool", level="ERROR"):
            alias.release()
            del alias
            gc.collect()
        assert pool._segs is not None
        self.assertEqual([segment.name for segment in pool._segs], names)
        with self.assertLogs("spdl.pipeline._arena._pool", level="ERROR"):
            pool.close()
        registration.close.side_effect = None
        pool.close()
        self.assertIsNone(pool._segs)
        self.assertEqual(pool._registrations, {})
        pool.close()
        self.assertEqual(registration.close.call_count, 3)

    def test_failed_registration_rolls_back_prior_mappings(self) -> None:
        """A later initialization failure unwinds every earlier registration once."""
        first, second = Mock(), Mock()
        second.register.side_effect = RuntimeError("register root")
        pool, _ = self._pool([first, second])
        with self.assertRaisesRegex(RuntimeError, "register root"):
            pool.open_reader().read_binary(0, 1)
        first.close.assert_called_once()
        second.close.assert_called_once()
        self.assertEqual(pool._registrations, {})

    def test_unregister_failure_quarantine_outlives_pool_and_last_alias(self) -> None:
        """Unknown unregister completion cannot unmap after the caller drops the pool."""
        registration = Mock()
        registration.close.side_effect = RuntimeError("unregister failure")
        pool, _ = self._pool([registration], cleanup=False)
        value = self._publish(pool)
        reference = weakref.ref(pool)
        pool.close()
        pool.unlink()
        del pool
        with self.assertLogs("spdl.pipeline._arena._pool", level="ERROR"):
            del value
            gc.collect()
        retained = reference()
        self.assertIsNotNone(retained)
        self.assertIsNotNone(retained._segs)
        registration.close.side_effect = None
        retained.close()
        self.assertIsNone(retained._segs)
        self.assertEqual(retained._registrations, {})
        del retained
        gc.collect()
        self.assertIsNone(reference())

    def test_interrupt_during_unregister_retains_mapping(self) -> None:
        """Control-flow interruption preserves independent mapping ownership until retry."""
        registration = Mock()
        pool, _ = self._pool([registration], cleanup=False)
        value = self._publish(pool)
        del value
        gc.collect()
        registration.close.side_effect = KeyboardInterrupt()
        reference = weakref.ref(pool)
        with self.assertRaises(KeyboardInterrupt):
            pool.close()
        pool.unlink()
        del pool
        gc.collect()
        retained = reference()
        self.assertIsNotNone(retained)
        self.assertIsNotNone(retained._segs)
        registration.close.side_effect = None
        retained.close()
        self.assertIsNone(retained._segs)
        del retained
        gc.collect()
        self.assertIsNone(reference())

    def test_initialization_root_survives_interrupted_rollback(self) -> None:
        """Rollback interruptions do not replace the initialization root or discard holds."""
        first, second = Mock(), Mock()
        second.register.side_effect = RuntimeError("register root")
        first.close.side_effect = KeyboardInterrupt()
        pool, _ = self._pool([first, second], cleanup=False)
        reference = weakref.ref(pool)
        with self.assertRaisesRegex(RuntimeError, "register root") as caught:
            pool.open_reader().read_binary(0, 1)
        self.assertIn("KeyboardInterrupt", caught.exception.__notes__[0])
        pool.unlink()
        del pool
        gc.collect()
        retained = reference()
        self.assertIsNotNone(retained)
        first.close.side_effect = None
        retained.close()
        self.assertIsNone(retained._segs)

    def test_ring_copies_seams_directly_to_independent_destination(self) -> None:
        """Pinned seams copy into independent allocator storage."""
        with patch(
            "spdl.pipeline._arena._ring._allocate_pinned",
            side_effect=lambda n: memoryview(bytearray(n)),
        ) as allocate:
            ring = SharedMemoryRingBuffer(16, acquire_timeout=0, pin_memory=True)
            self.addCleanup(ring.unlink)
            self.addCleanup(ring.close)
            writer, reader = ring.open_writer(), ring.open_reader()
            writer.begin_unit()
            writer.write_binary(b"x" * 13)
            reader.end_unit(writer.commit_unit(), [])
            writer.begin_unit()
            off, size = writer.write_binary(b"new data")
            span = writer.commit_unit()
            value = reader.read_binary(off, size)
            reader.end_unit(span, [])
            writer.begin_unit()
            writer.write_binary(b"z" * 16)
            writer.commit_unit()
            self.assertEqual(bytes(value), b"new data")
            allocate.assert_called_once_with(8)


class PinningDeviceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cuda = Mock()
        self.cuda.device_count.return_value = 2
        self.cuda._is_in_bad_fork.return_value = False
        self.device = object()
        self.torch = SimpleNamespace(
            cuda=self.cuda, device=Mock(return_value=self.device)
        )
        self.enterContext(patch.object(_pinning, "torch", self.torch))

    def test_local_rank_selects_consumer_device(self) -> None:
        """A valid LOCAL_RANK selects that device, with rank zero as the default."""
        for rank in (None, "1"):
            with self.subTest(rank=rank), patch.dict(os.environ, {}, clear=True):
                if rank is not None:
                    os.environ["LOCAL_RANK"] = rank
                self.assertIs(_pinning._local_device(), self.device)
                self.torch.device.assert_called_with("cuda", int(rank or "0"))

    def test_invalid_rank_is_rejected_before_device_selection(self) -> None:
        """Negative, out-of-range and unavailable device selections fail explicitly."""
        for rank, count in (("-1", 2), ("2", 2), ("0", 0)):
            with self.subTest(rank=rank, count=count):
                self.cuda.device_count.return_value = count
                with patch.dict(os.environ, {"LOCAL_RANK": rank}):
                    with self.assertRaisesRegex(RuntimeError, "CUDA device index"):
                        _pinning._local_device()
        self.torch.device.assert_not_called()

    def test_cuda_after_fork_and_noninteger_rank_fail(self) -> None:
        """Unsafe fork initialization and malformed rank values are not accepted."""
        self.cuda._is_in_bad_fork.return_value = True
        with patch.dict(os.environ, {"LOCAL_RANK": "0"}):
            with self.assertRaisesRegex(RuntimeError, "after fork; use spawn"):
                _pinning._local_device()
        with patch.dict(os.environ, {"LOCAL_RANK": "invalid"}):
            with self.assertRaises(ValueError):
                _pinning._local_device()
        self.torch.device.assert_not_called()


class PinningBudgetTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.enterContext(
            patch.object(
                _pinning,
                "Path",
                side_effect=lambda path: self.root / str(path).lstrip("/"),
            )
        )

    def _file(self, path: str, contents: str) -> None:
        target = self.root / path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)

    def test_shared_budget_checks_actual_tmpfs_free_bytes(self) -> None:
        """Shared allocations honor tmpfs free bytes; private outputs do not use tmpfs."""
        (self.root / "dev/shm").mkdir(parents=True)
        with patch.object(
            _pinning.os,
            "statvfs",
            return_value=SimpleNamespace(f_bavail=2, f_frsize=1024),
        ) as stat:
            _pinning._check_host_budget(2048, shared=True)
            with self.assertRaisesRegex(BufferError, "tmpfs has 2048 free"):
                _pinning._check_host_budget(2049, shared=True)
            stat.assert_called_with("/dev/shm")
            stat.reset_mock()
            _pinning._check_host_budget(4096, shared=False)
            stat.assert_not_called()

    def test_available_host_memory_is_checked_in_bytes(self) -> None:
        """MemAvailable kilobytes convert to bytes and include the equality boundary."""
        self._file("/proc/meminfo", "MemTotal: 9999 kB\nMemAvailable: 2 kB\n")
        _pinning._check_host_budget(2048, shared=False)
        with self.assertRaisesRegex(BufferError, "host has 2048 available"):
            _pinning._check_host_budget(2049, shared=False)

    def test_cgroup_limit_uses_remaining_headroom(self) -> None:
        """Cgroup-v2 current usage is subtracted from the finite limit."""
        self._file("/proc/self/cgroup", "7:cpu:/other\n0::/test-group\n")
        self._file("/sys/fs/cgroup/test-group/memory.max", "4096\n")
        self._file("/sys/fs/cgroup/test-group/memory.current", "3072\n")
        _pinning._check_host_budget(1024, shared=False)
        with self.assertRaisesRegex(BufferError, "cgroup headroom is 1024"):
            _pinning._check_host_budget(1025, shared=False)

    def test_unlimited_and_unavailable_budget_sources_are_not_zero(self) -> None:
        """Missing telemetry, non-v2 groups and max limits do not fabricate no capacity."""
        _pinning._check_host_budget(4096, shared=True)
        self._file("/proc/meminfo", "MemTotal: 1 kB\n")
        self._file("/proc/self/cgroup", "3:memory:/legacy\n")
        _pinning._check_host_budget(4096, shared=False)
        self._file("/proc/self/cgroup", "0::/test-group\n")
        _pinning._check_host_budget(4096, shared=False)
        self._file("/sys/fs/cgroup/test-group/memory.max", "max\n")
        self._file(
            "/sys/fs/cgroup/test-group/memory.current", "not-needed-for-unlimited\n"
        )
        _pinning._check_host_budget(4096, shared=False)


class HostRegistrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.storage = bytearray(64)
        self.runtime = Mock()
        self.runtime.cudaHostRegister.return_value = 0
        self.runtime.cudaHostUnregister.return_value = 0
        self.cuda = SimpleNamespace(
            device=Mock(side_effect=lambda device: nullcontext()),
            cudart=Mock(return_value=self.runtime),
        )
        self.enterContext(
            patch.object(_pinning, "torch", SimpleNamespace(cuda=self.cuda))
        )
        self.registration = _pinning._HostRegistration(
            memoryview(self.storage), "cuda:1"
        )
        self.addCleanup(self._close_registration)

    def _close_registration(self) -> None:
        self.runtime.cudaHostUnregister.side_effect = None
        self.runtime.cudaHostUnregister.return_value = 0
        self.registration.close()

    def test_full_mapping_registration_and_close_are_idempotent(self) -> None:
        """Register/unregister each whole mapping once, then release its retained export."""
        reg = self.registration
        reg.register()
        reg.register()
        self.runtime.cudaHostRegister.assert_called_once_with(reg.address, 64, 0)
        self.assertEqual(reg._state, "registered")
        with self.assertRaises(BufferError):
            self.storage.extend(b"x")
        reg.close()
        reg.close()
        self.runtime.cudaHostUnregister.assert_called_once_with(reg.address)
        self.assertEqual(reg._state, "unregistered")
        self.assertIsNone(reg._view)
        self.storage.extend(b"x")
        self.assertEqual(len(self.storage), 65)

    def test_reported_registration_error_leaves_no_registered_mapping(self) -> None:
        """An explicit CUDA error resets registration state and does not unregister twice."""
        self.runtime.cudaHostRegister.return_value = 2
        with self.assertRaisesRegex(
            RuntimeError, "cudaHostRegister failed with error 2"
        ):
            self.registration.register()
        self.assertEqual(self.registration._state, "unregistered")
        self.registration.close()
        self.runtime.cudaHostUnregister.assert_not_called()
        self.assertIsNone(self.registration._view)

    def test_registration_interrupt_keeps_uncertain_export_for_cleanup(self) -> None:
        """Unknown registration completion forbids repeat registration but retains backing."""
        error = KeyboardInterrupt("register interrupted")
        self.runtime.cudaHostRegister.side_effect = error
        with self.assertRaises(KeyboardInterrupt) as raised:
            self.registration.register()
        self.assertIs(raised.exception, error)
        self.assertEqual(self.registration._state, "uncertain")
        with self.assertRaisesRegex(RuntimeError, "completion is uncertain"):
            self.registration.register()
        self.runtime.cudaHostRegister.assert_called_once()
        with self.assertRaises(BufferError):
            self.storage.extend(b"x")
        self.registration.close()
        self.runtime.cudaHostUnregister.assert_called_once()
        self.assertIsNone(self.registration._view)

    def test_failed_unregister_retains_view_until_successful_retry(self) -> None:
        """Unregister errors and interruptions cannot release the mapping early."""
        self.registration.register()
        self.runtime.cudaHostUnregister.return_value = 3
        with self.assertRaisesRegex(RuntimeError, "mapping retained"):
            self.registration.close()
        self.assertIsNotNone(self.registration._view)
        self.assertEqual(self.registration._state, "registered")
        self.runtime.cudaHostUnregister.side_effect = KeyboardInterrupt(
            "unregister interrupted"
        )
        with self.assertRaises(KeyboardInterrupt):
            self.registration.close()
        self.assertIsNotNone(self.registration._view)
        self.runtime.cudaHostUnregister.side_effect = None
        self.runtime.cudaHostUnregister.return_value = 0
        self.registration.close()
        self.assertIsNone(self.registration._view)
        self.assertEqual(self.runtime.cudaHostUnregister.call_count, 3)

    def test_inherited_registration_is_rejected_without_cuda_calls(self) -> None:
        """An inherited mapping cannot register or unregister in another process."""
        with patch.object(
            _pinning.os, "getpid", return_value=self.registration._pid + 1
        ):
            with self.assertRaisesRegex(RuntimeError, "inherited across fork"):
                self.registration.register()
            with self.assertRaisesRegex(RuntimeError, "inherited across fork"):
                self.registration.close()
        self.runtime.cudaHostRegister.assert_not_called()
        self.runtime.cudaHostUnregister.assert_not_called()
        self.assertIsNotNone(self.registration._view)


class PinnedAllocationTest(unittest.TestCase):
    def test_negative_size_does_not_load_io_or_cuda(self) -> None:
        """Invalid sizes fail before optional native imports or device initialization."""
        with patch("importlib.import_module") as import_module:
            with self.assertRaisesRegex(ValueError, "non-negative"):
                _pinning._allocate_pinned(-1)
            import_module.assert_not_called()

    def test_pinned_allocation_exports_the_allocator_owner_without_copy(self) -> None:
        """The real native exporter retains the exact allocation through derived aliases."""

        class Storage:
            def __init__(self) -> None:
                self.data = bytearray(b"safe")

            def data_ptr(self) -> int:
                return ctypes.addressof(ctypes.c_char.from_buffer(self.data))

        storage = Storage()
        reference = weakref.ref(storage)
        empty = Mock(return_value=storage)
        cuda = SimpleNamespace(device=Mock(side_effect=lambda device: nullcontext()))
        torch = SimpleNamespace(empty=empty, uint8="uint8", cuda=cuda)
        with (
            patch.object(_pinning, "torch", torch),
            patch.object(_pinning, "_local_device", return_value="cuda:1"),
        ):
            view = _pinning._allocate_pinned(4)
        empty.assert_called_once_with(4, dtype="uint8", device="cpu", pin_memory=True)
        cuda.device.assert_called_once_with("cuda:1")
        alias = view[1:]
        view.release()
        del view
        empty.return_value = None
        del storage
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertEqual(bytes(alias), b"afe")
        alias[:1] = b"A"
        self.assertEqual(bytes(alias), b"Afe")
        alias.release()
        del alias
        gc.collect()
        self.assertIsNone(reference())
