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
        q = mp.get_context("spawn").Queue()
        q._wlock.acquire()
        q.put("tiny command")
        feeder = q._thread
        start = time.monotonic()
        try:
            with self.assertLogs(_LG, level="WARNING") as logs:
                _close_queue(q, abandon=False)
            self.assertTrue(feeder.is_alive())
            self.assertTrue(q._joincancelled)
            self.assertLess(time.monotonic() - start, 6.0)
            self.assertIn("Timed out joining", logs.output[0])
        finally:
            q._wlock.release()
            feeder.join(3.0)
            self.assertFalse(feeder.is_alive())

    def test_unclean_abandon_does_not_wait_for_real_writelock(self) -> None:
        q = mp.get_context("spawn").Queue()
        q._wlock.acquire()
        q.put("tiny command")
        feeder = q._thread
        start = time.monotonic()
        try:
            _close_queue(q, abandon=True)
            self.assertLess(time.monotonic() - start, 1.0)
            self.assertTrue(q._joincancelled)
        finally:
            q._wlock.release()
            feeder.join(3.0)
            self.assertFalse(feeder.is_alive())
