# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import multiprocessing as mp
import os
import unittest
from multiprocessing.synchronize import Event
from unittest.mock import Mock, patch

from spdl.pipeline import SharedMemoryRingBuffer, SharedMemorySegmentPool
from spdl.pipeline._arena import _sync


class _WindowsError(OSError):
    winerror: int


def _hold_writer(
    arena: SharedMemorySegmentPool | SharedMemoryRingBuffer,
    ready: Event,
    release: Event,
) -> None:
    writer = arena.open_writer()
    try:
        ready.set()
        if not release.wait(20):
            raise RuntimeError("writer release timed out")
    finally:
        del writer
        arena.close()


class ArenaProcessLivenessTest(unittest.TestCase):
    def test_spawned_writer_retains_storage_until_process_exit(self) -> None:
        """Both creators retain a live writer and retire only after its actual exit."""
        ctx = mp.get_context("spawn")
        for kind in ("pool", "ring"):
            with self.subTest(kind=kind):
                arena = (
                    SharedMemorySegmentPool(64, 1)
                    if kind == "pool"
                    else SharedMemoryRingBuffer(64)
                )
                ready, release = ctx.Event(), ctx.Event()
                process = ctx.Process(target=_hold_writer, args=(arena, ready, release))
                process.start()
                try:
                    self.assertTrue(ready.wait(10))
                    pid = process.pid
                    self.assertIsNotNone(pid)
                    if pid is None:
                        self.fail("Started process has no PID")
                    self.assertTrue(_sync._process_exists(pid))
                    arena.close()
                    self.assertFalse(arena._closed)
                    with self.assertRaisesRegex(BufferError, "retirement is pending"):
                        arena.unlink()
                    release.set()
                    process.join(10)
                    self.assertEqual(process.exitcode, 0)
                    self.assertFalse(_sync._process_exists(pid))
                    arena.close()
                    self.assertTrue(arena._closed)
                    arena.unlink()
                finally:
                    release.set()
                    process.join(10)
                    self.assertFalse(process.is_alive())
                    process.close()
                    arena.close()
                    arena.unlink()

    def test_current_process_and_unregistered_endpoint(self) -> None:
        """The current PID is live and an unregistered zero PID is absent."""
        self.assertTrue(_sync._process_exists(os.getpid()))
        self.assertFalse(_sync._process_exists(0))

    def test_windows_handle_wait_requires_positive_exit_evidence(self) -> None:
        """Only a signaled process HANDLE permits retirement; all waits close it."""
        kernel = Mock(SYNCHRONIZE=0x100000, WAIT_OBJECT_0=0)
        kernel.OpenProcess.return_value = 123
        for result, exists in ((0, False), (258, True), (0xFFFFFFFF, True)):
            with self.subTest(result=result):
                kernel.reset_mock()
                kernel.WaitForSingleObject.return_value = result
                with (
                    patch.object(_sync.os, "name", "nt"),
                    patch.dict("sys.modules", {"_winapi": kernel}),
                    patch.object(_sync.os, "kill") as kill,
                ):
                    self.assertEqual(_sync._process_exists(42), exists)
                kernel.OpenProcess.assert_called_once_with(0x100000, False, 42)
                kernel.WaitForSingleObject.assert_called_once_with(123, 0)
                kernel.CloseHandle.assert_called_once_with(123)
                kill.assert_not_called()

    def test_windows_missing_pid_and_uncertain_open_failures(self) -> None:
        """A missing Windows PID is absent; denied and unknown opens retain storage."""
        for winerror, exists in ((87, False), (5, True), (6, True)):
            with self.subTest(winerror=winerror):
                error = _WindowsError("OpenProcess failed")
                error.winerror = winerror
                kernel = Mock(SYNCHRONIZE=0x100000, WAIT_OBJECT_0=0)
                kernel.OpenProcess.side_effect = error
                with (
                    patch.object(_sync.os, "name", "nt"),
                    patch.dict("sys.modules", {"_winapi": kernel}),
                ):
                    self.assertEqual(_sync._process_exists(42), exists)
                kernel.WaitForSingleObject.assert_not_called()
                kernel.CloseHandle.assert_not_called()

    def test_windows_wait_or_handle_cleanup_failure_is_uncertain(self) -> None:
        """Wait and HANDLE cleanup failures preserve conservative liveness."""
        for operation in ("WaitForSingleObject", "CloseHandle"):
            with self.subTest(operation=operation):
                kernel = Mock(SYNCHRONIZE=0x100000, WAIT_OBJECT_0=0)
                kernel.OpenProcess.return_value = 123
                kernel.WaitForSingleObject.return_value = 0
                getattr(kernel, operation).side_effect = OSError("unknown HANDLE state")
                with (
                    patch.object(_sync.os, "name", "nt"),
                    patch.dict("sys.modules", {"_winapi": kernel}),
                ):
                    self.assertTrue(_sync._process_exists(42))
                kernel.CloseHandle.assert_called_once_with(123)

    def test_windows_invalid_pid_range_is_uncertain(self) -> None:
        """Out-of-range PIDs cannot masquerade as missing DWORD process IDs."""
        kernel = Mock()
        with (
            patch.object(_sync.os, "name", "nt"),
            patch.dict("sys.modules", {"_winapi": kernel}),
        ):
            self.assertTrue(_sync._process_exists(-1))
            self.assertTrue(_sync._process_exists(1 << 32))
        kernel.OpenProcess.assert_not_called()

    def test_posix_lookup_preserves_uncertainty(self) -> None:
        """POSIX absence and permission-denial behavior remains unchanged."""
        for error, exists in ((ProcessLookupError(), False), (PermissionError(), True)):
            with self.subTest(error=type(error).__name__):
                with (
                    patch.object(_sync.os, "name", "posix"),
                    patch.object(_sync.os, "kill", side_effect=error) as kill,
                ):
                    self.assertEqual(_sync._process_exists(42), exists)
                kill.assert_called_once_with(42, 0)
