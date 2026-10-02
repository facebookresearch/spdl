# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import multiprocessing as mp
import queue
import time
import unittest
from collections.abc import Iterable
from functools import partial
from typing import Any
from unittest.mock import call, MagicMock, patch

from spdl.pipeline import iterate_in_subprocess
from spdl.pipeline._iter_utils._common import _Cmd, _Msg, _Status


def _short_source() -> Iterable[int]:
    return range(1)


def _blocked_initializer(release: Any) -> None:
    release.wait()


class IterateInSubprocessLifecycleTest(unittest.TestCase):
    def test_discard_failure_does_not_skip_teardown(self) -> None:
        """A discarded payload failure does not skip worker and IPC cleanup."""
        errors = (
            RuntimeError("received 0 items of ancdata"),
            SystemExit("payload reducer exited"),
        )
        for error in errors:
            with self.subTest(error=type(error).__name__):
                cmd_q = MagicMock()
                data_q = MagicMock()
                data_q.get.return_value = _Msg(_Status.INITIALIZATION_SUCCEEDED)
                data_q.get_nowait.side_effect = error
                process = MagicMock()
                process.pid = 1
                process.is_alive.return_value = True
                process.exitcode = 0
                context = MagicMock()
                context.Queue.side_effect = [cmd_q, data_q]
                context.Process.return_value = process
                arena = MagicMock()

                with patch(
                    "spdl.pipeline._iter_utils._subprocess.mp.get_context",
                    return_value=context,
                ):
                    source = iterate_in_subprocess(_short_source, arena=arena)

                del source
                gc.collect()

                cmd_q.put.assert_called_once_with(_Cmd.ABORT)
                data_q.get_nowait.assert_called_once_with()
                process.join.assert_called_once_with(3)
                arena.shutdown_arena.assert_called_once_with()
                arena.close.assert_called_once_with()
                arena.unlink.assert_called_once_with()
                for ipc_queue in (cmd_q, data_q):
                    ipc_queue.cancel_join_thread.assert_called_once_with()
                    ipc_queue.close.assert_called_once_with()

    def test_start_failure_releases_ipc_resources(self) -> None:
        """Queue cleanup failures do not skip other owned resources."""
        cmd_q = MagicMock()
        cmd_q.cancel_join_thread.side_effect = SystemExit("queue cleanup failed")
        data_q = MagicMock()
        data_q.get_nowait.side_effect = queue.Empty
        cleanup = MagicMock()
        cleanup.attach_mock(cmd_q.cancel_join_thread, "cancel_cmd")
        cleanup.attach_mock(cmd_q.close, "close_cmd")
        cleanup.attach_mock(data_q.cancel_join_thread, "cancel_data")
        cleanup.attach_mock(data_q.close, "close_data")
        process = MagicMock()
        process.pid = None
        process.start.side_effect = RuntimeError("start failed")
        context = MagicMock()
        context.Queue.side_effect = [cmd_q, data_q]
        context.Process.return_value = process
        arena = MagicMock()

        with patch(
            "spdl.pipeline._iter_utils._subprocess.mp.get_context",
            return_value=context,
        ):
            with self.assertRaisesRegex(RuntimeError, "start failed"):
                iterate_in_subprocess(_short_source, arena=arena)

        process.join.assert_not_called()
        arena.shutdown_arena.assert_called_once_with()
        arena.close.assert_called_once_with()
        arena.unlink.assert_called_once_with()
        for ipc_queue in (cmd_q, data_q):
            ipc_queue.cancel_join_thread.assert_called_once_with()
            ipc_queue.close.assert_called_once_with()
            ipc_queue.join_thread.assert_not_called()
        self.assertEqual(
            cleanup.method_calls,
            [
                call.cancel_cmd(),
                call.close_cmd(),
                call.cancel_data(),
                call.close_data(),
            ],
        )

    def test_initializer_timeout_reaps_partially_started_worker(self) -> None:
        """A subprocess that misses initialization timeout is reaped immediately."""
        before = {process.pid for process in mp.active_children()}
        release = mp.get_context("spawn").Event()

        try:
            t0 = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "did not initialize"):
                iterate_in_subprocess(
                    _short_source,
                    initializer=partial(_blocked_initializer, release),
                    mp_context="spawn",
                    timeout=0.05,
                )
            self.assertLess(time.monotonic() - t0, 2.0)

            after = {process.pid for process in mp.active_children()}
            self.assertEqual(after - before, set())
        finally:
            # Let a leaked worker from the pre-fix implementation terminate instead of
            # carrying a failed test's subprocess into the rest of the suite.
            release.set()
            for process in mp.active_children():
                if process.pid not in before:
                    process.join(timeout=1)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=1)
