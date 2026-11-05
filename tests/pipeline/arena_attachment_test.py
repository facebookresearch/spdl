# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest
from multiprocessing import resource_tracker, shared_memory
from unittest.mock import patch

from spdl.pipeline._arena._shm import _attach, _UntrackedSharedMemory


class ArenaAttachmentTest(unittest.TestCase):
    def test_attachment_does_not_mutate_the_owner_tracker(self) -> None:
        """Consumer attachments neither register nor unregister shared owner names."""
        owner = shared_memory.SharedMemory(create=True, size=64)
        self.addCleanup(owner.unlink)
        self.addCleanup(owner.close)
        with (
            patch.object(resource_tracker, "register") as register,
            patch.object(resource_tracker, "unregister") as unregister,
        ):
            reader = _attach(owner.name)
            self.addCleanup(reader.close)
            reader_buf, owner_buf = reader.buf, owner.buf
            if reader_buf is None or owner_buf is None:
                self.fail("Mapping has no buffer")
            reader_buf[0] = 71
            self.assertEqual(owner_buf[0], 71)
            register.assert_not_called()
            unregister.assert_not_called()

    @unittest.skipIf(sys.platform == "win32", "POSIX fallback only")
    def test_stock_pre_track_fallback_preserves_ownership(self) -> None:
        """Stock older CPython fallback attaches without any tracker operation."""
        owner = shared_memory.SharedMemory(create=True, size=64)
        self.addCleanup(owner.unlink)
        self.addCleanup(owner.close)
        with (
            patch.object(resource_tracker, "register") as register,
            patch.object(resource_tracker, "unregister") as unregister,
        ):
            reader = _UntrackedSharedMemory(owner.name)
            self.addCleanup(reader.close)
            reader_buf, owner_buf = reader.buf, owner.buf
            if reader_buf is None or owner_buf is None:
                self.fail("Mapping has no buffer")
            reader_buf[0] = 82
            self.assertEqual(owner_buf[0], 82)
            register.assert_not_called()
            unregister.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "Non-owning"):
                reader.unlink()

    @unittest.skipIf(sys.platform == "win32", "POSIX mapping failure only")
    def test_mapping_failure_does_not_unlink_owner_and_closes_fd(self) -> None:
        """An attachment mapping failure preserves the live name and closes its opened fd."""
        owner = shared_memory.SharedMemory(create=True, size=64)
        self.addCleanup(owner.unlink)
        self.addCleanup(owner.close)
        with (
            patch(
                "spdl.pipeline._arena._shm.mmap.mmap", side_effect=OSError("mmap root")
            ),
            patch("spdl.pipeline._arena._shm.os.close", wraps=os.close) as close,
        ):
            with self.assertRaisesRegex(OSError, "mmap root"):
                _attach(owner.name)
        close.assert_called_once()
        with self.assertRaises(OSError):
            os.fstat(close.call_args.args[0])
        reader = _attach(owner.name)
        reader.close()
        self.assertEqual(owner.size, 64)

    def test_public_default_arenas_support_spawn_and_clean_owner_unlink(self) -> None:
        """Default arenas support spawn and clean unlink without private flags."""
        script = textwrap.dedent(
            """
            from functools import partial
            from multiprocessing import shared_memory
            from spdl.pipeline import (
                iterate_in_subprocess, SharedMemoryRingBuffer, SharedMemorySegmentPool,
            )

            if __name__ == '__main__':
                for backend in ('pool', 'ring'):
                    arena = (SharedMemorySegmentPool(32768, 2) if backend == 'pool'
                             else SharedMemoryRingBuffer(65536))
                    names = ([arena._ctrl_name, *arena._seg_names] if backend == 'pool'
                             else [arena._shm_name])
                    inputs = [bytes([i]) * 8192 for i in range(8)]
                    got = []
                    for item in iterate_in_subprocess(partial(list, inputs), arena=arena,
                                                       mp_context='spawn', buffer_size=1):
                        got.append(bytes(item))
                        del item
                    assert got == inputs
                    arena.close()
                    arena.unlink()
                    arena.close()
                    arena.unlink()
                    for name in names:
                        try:
                            handle = shared_memory.SharedMemory(name=name)
                        except FileNotFoundError:
                            continue
                        handle.close()
                        handle.unlink()
                        raise AssertionError('name still exists: ' + name)
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("resource_tracker", result.stderr, result.stderr)
        self.assertNotIn("BufferError", result.stderr, result.stderr)
