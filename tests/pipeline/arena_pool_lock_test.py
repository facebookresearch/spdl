# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import struct
import subprocess
import sys
import threading
import time
import unittest
from collections.abc import Callable
from unittest.mock import Mock, patch

from spdl.pipeline import SharedMemorySegmentPool
from spdl.pipeline._arena import _pool


def _holds_control_lock(pool: SharedMemorySegmentPool) -> bool:
    return pool._space_cv._lock._semlock._is_mine()  # type: ignore[attr-defined]


class _RecordingSegment:
    """A segment view that records whether each write holds the control lock."""

    def __init__(self, pool: SharedMemorySegmentPool, view: memoryview) -> None:
        self._pool = pool
        self._view = view
        self.writes_under_lock: list[bool] = []

    def __setitem__(self, key: slice, value: memoryview) -> None:
        self.writes_under_lock.append(_holds_control_lock(self._pool))
        self._view[key] = value


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


class ArenaPoolLockTest(unittest.TestCase):
    def _pool(self, segment_size: int, count: int) -> SharedMemorySegmentPool:
        pool = SharedMemorySegmentPool(segment_size, count, acquire_timeout=0)
        self.addCleanup(pool.unlink)
        self.addCleanup(pool.close)
        return pool

    def _hold_lock(
        self, pool: SharedMemorySegmentPool, seconds: float
    ) -> threading.Thread:
        held = threading.Event()

        def hold() -> None:
            with pool._space_cv:
                held.set()
                time.sleep(seconds)

        thread = threading.Thread(target=hold)
        thread.start()
        self.addCleanup(thread.join, 10)
        self.assertTrue(held.wait(10))
        return thread

    def test_unit_copy_runs_outside_the_control_lock(self) -> None:
        """A writer copies a unit into its segment without holding the shared lock."""
        pool = self._pool(1 << 20, 2)
        writer, reader = pool.open_writer(), pool.open_reader()
        writer.begin_unit()
        segment = _RecordingSegment(pool, pool._segment(0))
        with patch.object(pool, "_segment", return_value=segment):
            offset, size = writer.write_binary(b"x" * 4096)
        span = writer.commit_unit()
        self.assertEqual(segment.writes_under_lock, [False])
        self.assertEqual(bytes(reader.read_binary(offset, size)), b"x" * 4096)
        reader.end_unit(span)

    def test_first_pinned_read_registers_outside_the_control_lock(self) -> None:
        """A pinned consumer registers its segments without holding the shared lock."""
        registration = Mock()
        registered_under_lock: list[bool] = []
        self.enterContext(
            patch.object(_pool, "_HostRegistration", return_value=registration)
        )
        self.enterContext(patch.object(_pool, "_local_device", return_value="cuda:0"))
        self.enterContext(patch.object(_pool, "_check_host_budget"))
        pool = SharedMemorySegmentPool(64, 1, acquire_timeout=0, pin_memory=True)
        self.addCleanup(pool.unlink)
        self.addCleanup(pool.close)
        registration.register.side_effect = lambda: registered_under_lock.append(
            _holds_control_lock(pool)
        )
        writer, reader = pool.open_writer(), pool.open_reader()
        writer.begin_unit()
        offset, size = writer.write_binary(b"a" * 64)
        span = writer.commit_unit()
        view = reader.read_binary(offset, size)
        self.assertEqual(registered_under_lock, [False])
        view.release()
        reader.end_unit(span)

    def test_slow_lock_holder_does_not_retire_the_pool(self) -> None:
        """A unit transfer outwaits a slow lock holder while every endpoint lives."""
        pool = self._pool(64, 2)
        writer = pool.open_writer()
        pool.open_reader()
        writer.begin_unit()
        holder = self._hold_lock(pool, 0.5)
        with (
            patch.object(_pool, "_CONTROL_LOCK_TIMEOUT", 0.05),
            self.assertLogs(_pool._LG, "WARNING") as logs,
        ):
            self.assertEqual(writer.write_binary(b"live"), (0, 4))
        holder.join(10)
        self.assertIsNone(pool._retirement_error)
        self.assertIn("arena control lock", logs.output[0])
        writer.commit_unit()

    def test_lock_timeout_with_a_dead_endpoint_still_retires_the_pool(self) -> None:
        """A lock wait whose other endpoint process is gone retires the pool."""
        pool = self._pool(64, 2)
        writer = pool.open_writer()
        writer.begin_unit()
        # The consumer endpoint slot of the control block (<QQBBQQQ>, offset 26).
        struct.pack_into("<Q", pool._ctrl_view(), 26, _dead_pid())
        holder = self._hold_lock(pool, 0.5)
        try:
            with patch.object(_pool, "_CONTROL_LOCK_TIMEOUT", 0.05):
                with self.assertRaisesRegex(RuntimeError, "control lock unavailable"):
                    writer.write_binary(b"dead")
            self.assertIsNotNone(pool._retirement_error)
        finally:
            holder.join(10)
            # The test-owned lock holder is known healthy again; only the test
            # clears the retirement state to let cleanup close the pool.
            pool._retirement_error = None

    @unittest.skipUnless(sys.platform == "linux", "minor-fault counts are Linux-only")
    def test_construction_faults_in_every_segment_page(self) -> None:
        """The first unit copied into a new pool takes far fewer page faults than a
        copy into fresh shared memory of the same size."""
        import resource
        from multiprocessing import shared_memory

        def faults_while(write: Callable[[], object]) -> int:
            before = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
            write()
            return resource.getrusage(resource.RUSAGE_SELF).ru_minflt - before

        size = 64 * 2**20
        data = b"\x01" * size
        fresh = shared_memory.SharedMemory(create=True, size=size)
        self.addCleanup(fresh.unlink)
        self.addCleanup(fresh.close)
        fresh_view = fresh.buf
        assert fresh_view is not None
        fresh_faults = faults_while(lambda: fresh_view.__setitem__(slice(None), data))
        pool = self._pool(size, 1)
        writer = pool.open_writer()
        pool.open_reader()
        writer.begin_unit()
        pool_faults = faults_while(lambda: writer.write_binary(data))
        writer.commit_unit()
        # Fresh shared memory faults once per page (4 KiB, or 2 MiB with shmem THP).
        self.assertGreater(fresh_faults, 8)
        self.assertLess(pool_faults, fresh_faults // 4)
