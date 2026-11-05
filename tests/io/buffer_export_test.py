# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native buffer ownership checks independent of pipeline arena accounting."""

import ctypes
import gc
import importlib
import unittest
import weakref


class BufferExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.buffer = importlib.import_module("spdl.io.lib._buffer")

    def test_legacy_export_notifies_after_last_alias(self) -> None:
        """A legacy release callback follows the final native buffer alias."""
        calls = []
        view = self.buffer._export(
            memoryview(bytearray(b"abcd")), lambda: calls.append(1)
        )
        alias = view[1:]
        view.release()
        del view
        gc.collect()
        self.assertEqual(calls, [])
        self.assertEqual(bytes(alias), b"bcd")
        alias.release()
        del alias
        gc.collect()
        self.assertEqual(calls, [1])

    def test_native_adoption_prevents_setup_rollback(self) -> None:
        """Python rollback cannot retire an adopted view while its storage is live."""
        calls = []
        ticket = self.buffer._release_ticket(lambda: calls.append(1))
        ticket._activate()
        view = self.buffer._export(memoryview(bytearray(b"safe")), ticket)
        ticket._rollback()
        self.assertEqual(calls, [])
        self.assertEqual(bytes(view), b"safe")
        view.release()
        del view
        gc.collect()
        ticket._rollback()
        self.assertEqual(calls, [1])

    def test_setup_rollback_notifies_once(self) -> None:
        """Repeated setup rollback retires the acquired bookkeeping hold once."""
        calls = []
        ticket = self.buffer._release_ticket(lambda: calls.append(1))
        ticket._activate()
        ticket._rollback()
        ticket._rollback()
        self.assertEqual(calls, [1])

    def test_cancel_does_not_notify(self) -> None:
        """Canceling an inactive setup ticket does not release an unacquired hold."""
        calls = []
        ticket = self.buffer._release_ticket(lambda: calls.append(1))
        ticket._cancel()
        ticket._rollback()
        self.assertEqual(calls, [])

    def test_owned_export_retains_allocator_until_last_alias(self) -> None:
        """A native pointer export keeps its allocator owner through the last alias."""

        class Owner:
            def __init__(self) -> None:
                self.storage = ctypes.create_string_buffer(b"owned")

        owner = Owner()
        reference = weakref.ref(owner)
        view = self.buffer._export_owned(ctypes.addressof(owner.storage), 5, owner)
        alias = view[1:]
        del owner, view
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertEqual(bytes(alias), b"wned")
        alias.release()
        del alias
        gc.collect()
        self.assertIsNone(reference())

    def test_empty_export_releases_once(self) -> None:
        """An empty native export still owns and retires one release callback."""
        calls = []
        view = self.buffer._export(memoryview(bytearray()), lambda: calls.append(1))
        self.assertEqual(bytes(view), b"")
        view.release()
        del view
        gc.collect()
        self.assertEqual(calls, [1])
