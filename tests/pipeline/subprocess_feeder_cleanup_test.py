# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from spdl.pipeline._iter_utils import _subprocess as m


class SubprocessFeederCleanupTest(unittest.TestCase):
    def test_normal_real_queue_feeder_is_joined(self) -> None:
        """A normal queue close joins its real feeder thread."""
        q = mp.get_context("spawn").Queue()
        q.put("tiny command")
        feeder = q._thread  # type: ignore[attr-defined]
        m._close_queue(q)
        self.assertFalse(feeder.is_alive())

    def test_blocked_feeder_wait_is_bounded_and_abandoned(self) -> None:
        """A blocked feeder is abandoned after a bounded wait."""
        gate = threading.Event()
        feeder = threading.Thread(target=gate.wait, daemon=True)
        feeder.start()
        q = SimpleNamespace(
            _thread=feeder, close=Mock(), join_thread=Mock(), cancel_join_thread=Mock()
        )
        try:
            with self.assertLogs(m._LG, level="WARNING") as logs:
                close_queue = m._close_queue
                start = time.monotonic()
                close_queue(q)
                elapsed = time.monotonic() - start
            self.assertLess(elapsed, 6.0)
            q.cancel_join_thread.assert_called_once()
            q.join_thread.assert_not_called()
            self.assertIn("Timed out joining", logs.output[0])
        finally:
            gate.set()
            feeder.join(3)

    def test_clean_status_is_evaluated_after_worker_join(self) -> None:
        """Command cleanup observes the exit status established by worker join."""
        process = SimpleNamespace(exitcode=None)
        process.join = lambda timeout: setattr(process, "exitcode", 0)
        cmd_q, data_q = object(), object()
        interface = SimpleNamespace(process=process, cmd_q=cmd_q, data_q=data_q)
        with patch.object(m, "_close_queue") as close:
            with m._IPCResourceCleanup(
                interface,  # type: ignore[arg-type]
                cmd_q,
                data_q,
                None,
                process_started=True,
            ):
                pass
        self.assertEqual(
            [call.args[0] for call in close.call_args_list], [cmd_q, data_q]
        )
        self.assertEqual(
            [call.kwargs["abandon"] for call in close.call_args_list], [False, True]
        )
        self.assertIsNone(interface.cmd_q)
        self.assertIsNone(interface.data_q)

    def test_force_failed_and_unstarted_paths_stay_abandoned(self) -> None:
        """Forced, failed and unstarted workers retain queue abandonment."""
        for force, started, exitcode in [
            (True, True, 0),
            (False, True, -15),
            (False, False, 0),
        ]:
            with self.subTest(force=force, started=started, exitcode=exitcode):
                process = SimpleNamespace(exitcode=exitcode, join=lambda timeout: None)
                interface = SimpleNamespace(
                    process=process, cmd_q=object(), data_q=object()
                )
                with patch.object(m, "_close_queue") as close:
                    with m._IPCResourceCleanup(
                        interface,  # type: ignore[arg-type]
                        interface.cmd_q,
                        interface.data_q,
                        None,
                        process_started=started,
                        force=force,
                    ):
                        pass
                self.assertEqual(
                    [call.kwargs["abandon"] for call in close.call_args_list],
                    [True, True],
                )

    def test_active_primary_failure_stays_abandoned(self) -> None:
        """An already active failure preserves command-queue abandonment."""
        process = SimpleNamespace(exitcode=0, join=lambda timeout: None)
        interface = SimpleNamespace(process=process, cmd_q=object(), data_q=object())
        try:
            raise RuntimeError("primary")
        except RuntimeError:
            cleanup = m._IPCResourceCleanup(
                interface,  # type: ignore[arg-type]
                interface.cmd_q,
                interface.data_q,
                None,
                process_started=True,
            )
        with patch.object(m, "_close_queue") as close:
            with cleanup:
                pass
        self.assertEqual(
            [call.kwargs["abandon"] for call in close.call_args_list], [True, True]
        )

    def test_body_failure_stays_abandoned_and_is_not_suppressed(self) -> None:
        """A body failure keeps queue abandonment and propagates unchanged."""
        process = SimpleNamespace(exitcode=0, join=lambda timeout: None)
        interface = SimpleNamespace(process=process, cmd_q=object(), data_q=object())
        error = RuntimeError("body failure")
        with patch.object(m, "_close_queue") as close:
            with self.assertRaises(RuntimeError) as raised:
                with m._IPCResourceCleanup(
                    interface,  # type: ignore[arg-type]
                    interface.cmd_q,
                    interface.data_q,
                    None,
                    process_started=True,
                ):
                    raise error
        self.assertIs(raised.exception, error)
        self.assertEqual(
            [call.kwargs["abandon"] for call in close.call_args_list], [True, True]
        )
        self.assertIsNone(interface.cmd_q)
        self.assertIsNone(interface.data_q)

    def test_retirement_failure_stays_abandoned_and_is_not_suppressed(self) -> None:
        """Arena retirement failures abandon queues without masking the error."""
        for phase in ("close", "unlink"):
            with self.subTest(phase=phase):
                process = SimpleNamespace(exitcode=0, join=lambda timeout: None)
                interface = SimpleNamespace(
                    process=process, cmd_q=object(), data_q=object()
                )
                error = RuntimeError(phase + " failure")
                arena = SimpleNamespace(close=Mock(), unlink=Mock())
                getattr(arena, phase).side_effect = error
                with patch.object(m, "_close_queue") as close:
                    with self.assertRaises(RuntimeError) as raised:
                        with m._IPCResourceCleanup(
                            interface,  # type: ignore[arg-type]
                            interface.cmd_q,
                            interface.data_q,
                            arena,
                            process_started=True,
                        ):
                            pass
                self.assertIs(raised.exception, error)
                self.assertEqual(
                    [call.kwargs["abandon"] for call in close.call_args_list],
                    [True, True],
                )
                self.assertIsNone(interface.cmd_q)
                self.assertIsNone(interface.data_q)
