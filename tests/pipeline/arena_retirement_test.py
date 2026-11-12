# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import threading
import unittest
import weakref
from typing import Any
from unittest.mock import patch

from parameterized import parameterized
from spdl.pipeline import SharedMemoryRingBuffer, SharedMemorySegmentPool
from spdl.pipeline._arena import _pool, _ring


class ArenaRetirementTest(unittest.TestCase):
    @parameterized.expand([("pool",), ("ring",)])
    def test_contended_close_reports_failure_and_retains_owned_storage(
        self, kind: str
    ) -> None:
        """A bounded close failure keeps mappings and live aliases until safe retry."""
        arena: Any = (
            SharedMemorySegmentPool(64, 1, acquire_timeout=0)
            if kind == "pool"
            else SharedMemoryRingBuffer(64, acquire_timeout=0)
        )
        writer, reader = arena.open_writer(), arena.open_reader()
        writer.begin_unit()
        offset, size = writer.write_binary(b"safe" * 16)
        span = writer.commit_unit()
        view = reader.read_binary(offset, size)
        reader.end_unit(span)
        alias = view[1:]
        view.release()
        held, unlock = threading.Event(), threading.Event()

        def own_lock() -> None:
            with arena._space_cv:
                held.set()
                unlock.wait(10)

        thread = threading.Thread(target=own_lock)
        thread.start()
        module = _pool if kind == "pool" else _ring
        try:
            self.assertTrue(held.wait(10))
            with patch.object(module, "_CONTROL_LOCK_TIMEOUT", 0.05):
                with self.assertRaises(RuntimeError) as caught:
                    arena.close()
            self.assertIs(arena._retirement_error, caught.exception)
            self.assertFalse(arena._closed)
            self.assertEqual(bytes(alias), (b"safe" * 16)[1:])
            roots = (
                _pool._REGISTRATION_QUARANTINE
                if kind == "pool"
                else _ring._RETIREMENT_ROOTS
            )
            self.assertIn(arena, roots)
            reference = weakref.ref(arena)
            del writer, reader
            self.assertIs(reference(), arena)
        finally:
            unlock.set()
            thread.join(10)
            self.assertFalse(thread.is_alive())
            # The injected condition was healthy after this test-owned thread
            # released it. No production failure path clears retirement state.
            arena._retirement_error = None
            alias.release()
            arena.close()
            arena.unlink()

    @parameterized.expand([("pool",), ("ring",)])
    def test_repeated_completed_close_does_not_reaccess_released_control(
        self, kind: str
    ) -> None:
        """Already-retired handles close repeatedly without reopening control storage."""
        arena: Any = (
            SharedMemorySegmentPool(64, 1)
            if kind == "pool"
            else SharedMemoryRingBuffer(64)
        )
        try:
            arena.close()
            with patch.object(
                arena,
                "shutdown_arena",
                side_effect=AssertionError("released control"),
            ):
                arena.close()
                arena.close()
            self.assertTrue(arena._closed)
        finally:
            arena.unlink()
