# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import os
import time
import unittest

from spdl.pipeline._iter_utils._subprocess import _close_queue, _LG


@unittest.skipIf(os.name == "nt", "POSIX queue writer locks only")
class SubprocessFeederWriterCleanupTest(unittest.TestCase):
    def test_bounded_clean_join_with_real_abandoned_writelock(self) -> None:
        """A held writer lock cannot make clean queue teardown wait forever."""
        q = mp.get_context("spawn").Queue()
        q._wlock.acquire()  # type: ignore[attr-defined]
        q.put("tiny command")
        feeder = q._thread  # type: ignore[attr-defined]
        try:
            with self.assertLogs(_LG, level="WARNING") as logs:
                close_queue = _close_queue
                start = time.monotonic()
                close_queue(q, abandon=False)
                elapsed = time.monotonic() - start
            self.assertTrue(feeder.is_alive())
            self.assertTrue(q._joincancelled)  # type: ignore[attr-defined]
            self.assertLess(elapsed, 6.0)
            self.assertIn("Timed out joining", logs.output[0])
        finally:
            q._wlock.release()  # type: ignore[attr-defined]
            feeder.join(3.0)
            self.assertFalse(feeder.is_alive())

    def test_unclean_abandon_does_not_wait_for_real_writelock(self) -> None:
        """Unclean queue teardown does not wait on a held writer lock."""
        q = mp.get_context("spawn").Queue()
        q._wlock.acquire()  # type: ignore[attr-defined]
        q.put("tiny command")
        feeder = q._thread  # type: ignore[attr-defined]
        close_queue = _close_queue
        start = time.monotonic()
        try:
            close_queue(q, abandon=True)
            self.assertLess(time.monotonic() - start, 1.0)
            self.assertTrue(q._joincancelled)  # type: ignore[attr-defined]
        finally:
            q._wlock.release()  # type: ignore[attr-defined]
            feeder.join(3.0)
            self.assertFalse(feeder.is_alive())
