# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import unittest
from collections.abc import Iterable, Iterator
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from spdl.pipeline import iterate_in_subinterpreter, iterate_in_subprocess
from spdl.pipeline._iter_utils._common import (
    _get_worker_message,
    _Msg,
    _Status,
)
from spdl.pipeline._iter_utils._subprocess import _ipc


def _short_source() -> Iterable[int]:
    return range(1)


def _failing_source() -> Iterable[int]:
    yield from ()
    raise RuntimeError("worker iteration failed")


def _blocked_initializer(release: Any) -> None:
    release.wait()


def _cleanup_resources(process: Any, ipc_queues: tuple[Any, Any]) -> None:
    if process.pid is not None:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
    for ipc_queue in ipc_queues:
        try:
            ipc_queue.cancel_join_thread()
        except (OSError, ValueError):
            pass
        try:
            ipc_queue.close()
        except (OSError, ValueError):
            pass


def _recording_context(
    ctx: Any,
    ipc_queues: tuple[Any, Any],
    prepared_process: Any | None = None,
) -> tuple[Any, list[Any]]:
    queues = iter(ipc_queues)
    processes: list[Any] = []

    def make_queue(*args: Any, **kwargs: Any) -> Any:
        try:
            return next(queues)
        except StopIteration as error:
            raise AssertionError("unexpected extra Queue construction") from error

    def make_process(**kwargs: Any) -> Any:
        process = (
            prepared_process if prepared_process is not None else ctx.Process(**kwargs)
        )
        processes.append(process)
        return process

    return SimpleNamespace(Queue=make_queue, Process=make_process), processes


def _join_queue_thread(ipc_queue: Any, timeout: float = 1.0) -> None:
    """Join a queue feeder without letting an assertion hang the test process."""
    done = threading.Event()
    errors: list[BaseException] = []

    def join() -> None:
        try:
            ipc_queue.join_thread()
        except Exception as error:
            errors.append(error)
        finally:
            done.set()

    threading.Thread(target=join, daemon=True).start()
    if not done.wait(timeout):
        raise AssertionError("multiprocessing queue feeder did not terminate")
    if errors:
        raise errors[0]


class _AbruptProcessExitIterable:
    def __init__(self, release: Any) -> None:
        self._release = release

    def __iter__(self) -> Iterator[int]:
        yield 0
        # The parent releases this only after receiving the first result, so the queue
        # feeder has flushed all preceding protocol messages before abnormal teardown.
        self._release.wait()
        os._exit(17)


def _abrupt_process_exit_source(release: Any) -> Iterable[int]:
    return _AbruptProcessExitIterable(release)


class IterateInSubprocessLifecycleTest(unittest.TestCase):
    def _make_real_context(
        self,
    ) -> tuple[Any, tuple[Any, Any], list[Any]]:
        ctx = mp.get_context("spawn")
        cmd_q = ctx.Queue()
        data_q = ctx.Queue(maxsize=3)
        ipc_queues = (cmd_q, data_q)
        context, processes = _recording_context(ctx, ipc_queues)
        return context, ipc_queues, processes

    def _start_source_with_real_ipc(
        self,
    ) -> tuple[Iterable[int], Any, tuple[Any, Any]]:
        context, ipc_queues, processes = self._make_real_context()
        with patch(
            "spdl.pipeline._iter_utils._subprocess.mp.get_context",
            return_value=context,
        ):
            source = iterate_in_subprocess(_failing_source)
        self.assertEqual(len(processes), 1)
        process = processes[0]
        self.addCleanup(_cleanup_resources, process, ipc_queues)
        return source, process, ipc_queues

    def _assert_real_resources_released(
        self, process: Any, ipc_queues: tuple[Any, Any]
    ) -> None:
        self.assertFalse(process.is_alive())
        self.assertEqual(process.exitcode, 0)
        for ipc_queue in ipc_queues:
            with self.assertRaisesRegex(ValueError, "closed"):
                ipc_queue.put_nowait(None)
            _join_queue_thread(ipc_queue)

    def test_arena_shutdown_failure_does_not_mask_iteration_error(self) -> None:
        """Arena wakeup failure cannot skip teardown or replace iteration failure."""
        process = MagicMock(pid=1, exitcode=0)
        process.is_alive.return_value = False
        cmd_q = MagicMock()
        data_q = MagicMock()
        arena = MagicMock()
        arena.shutdown_arena.side_effect = RuntimeError("arena shutdown failed")
        interface = _ipc(process, cmd_q, data_q, 1.0, arena)

        with (
            patch("spdl.pipeline._iter_utils._subprocess._drain") as drain,
            self.assertLogs("spdl.pipeline._iter_utils._subprocess", level="WARNING"),
            self.assertRaisesRegex(RuntimeError, "iteration failed"),
        ):
            try:
                raise RuntimeError("iteration failed")
            finally:
                interface.terminate()

        self.assertEqual(drain.call_args_list, [((data_q,), {}), ((data_q,), {})])
        process.join.assert_called_once_with(3)
        arena.close.assert_called_once_with()
        arena.unlink.assert_called_once_with()
        for ipc_queue in (cmd_q, data_q):
            ipc_queue.cancel_join_thread.assert_called_once_with()
            ipc_queue.close.assert_called_once_with()

    def test_abort_send_failure_does_not_skip_teardown(self) -> None:
        """A broken command queue cannot skip arena wakeup or IPC cleanup."""
        process = MagicMock(pid=1, exitcode=0)
        process.is_alive.return_value = True
        cmd_q = MagicMock()
        cmd_q.put_nowait.side_effect = OSError("broken command queue")
        data_q = MagicMock()
        arena = MagicMock()
        interface = _ipc(process, cmd_q, data_q, 1.0, arena)

        with (
            patch("spdl.pipeline._iter_utils._subprocess._drain") as drain,
            self.assertLogs("spdl.pipeline._iter_utils._subprocess", level="DEBUG"),
        ):
            interface.terminate()

        arena.shutdown_arena.assert_called_once_with()
        self.assertEqual(drain.call_args_list, [((data_q,), {}), ((data_q,), {})])
        process.join.assert_called_once_with(3)
        arena.close.assert_called_once_with()
        arena.unlink.assert_called_once_with()
        for ipc_queue in (cmd_q, data_q):
            ipc_queue.cancel_join_thread.assert_called_once_with()
            ipc_queue.close.assert_called_once_with()

    def test_join_failure_does_not_skip_ipc_cleanup(self) -> None:
        """A process-reaping error still releases the arena and queue resources."""
        process = MagicMock(pid=1)
        process.is_alive.return_value = False
        cmd_q = MagicMock()
        data_q = MagicMock()
        arena = MagicMock()
        interface = _ipc(process, cmd_q, data_q, 1.0, arena)

        with (
            patch("spdl.pipeline._iter_utils._subprocess._drain") as drain,
            patch(
                "spdl.pipeline._iter_utils._subprocess._join",
                side_effect=RuntimeError("join failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "join failed"),
        ):
            interface.terminate()

        drain.assert_called_once_with(data_q)
        arena.close.assert_called_once_with()
        arena.unlink.assert_called_once_with()
        for ipc_queue in (cmd_q, data_q):
            ipc_queue.cancel_join_thread.assert_called_once_with()
            ipc_queue.close.assert_called_once_with()

    def test_stale_discard_failure_does_not_skip_real_teardown(self) -> None:
        """A stale payload error still reaps a real worker and closes its queues."""
        source, process, ipc_queues = self._start_source_with_real_ipc()

        with (
            patch(
                "spdl.pipeline._iter_utils._subprocess._drain",
                side_effect=RuntimeError("received 0 items of ancdata"),
            ),
            self.assertRaisesRegex(RuntimeError, "worker iteration failed"),
        ):
            list(source)

        self._assert_real_resources_released(process, ipc_queues)

    def test_control_flow_during_discard_propagates_after_teardown(self) -> None:
        """Process control-flow exceptions propagate after real IPC cleanup."""
        source, process, ipc_queues = self._start_source_with_real_ipc()

        with (
            patch(
                "spdl.pipeline._iter_utils._subprocess._drain",
                side_effect=SystemExit("payload reducer exited"),
            ),
            self.assertRaisesRegex(SystemExit, "payload reducer exited"),
        ):
            list(source)

        self._assert_real_resources_released(process, ipc_queues)

    def test_start_failure_releases_ipc_resources(self) -> None:
        """A start failure closes the real multiprocessing queues it allocated."""
        ctx = mp.get_context("spawn")
        cmd_q = ctx.Queue()
        data_q = ctx.Queue(maxsize=3)
        process = ctx.Process(target=_short_source)
        ipc_queues = (cmd_q, data_q)
        context, _ = _recording_context(ctx, ipc_queues, process)
        self.addCleanup(_cleanup_resources, process, ipc_queues)

        with (
            patch(
                "spdl.pipeline._iter_utils._subprocess.mp.get_context",
                return_value=context,
            ),
            patch.object(process, "start", side_effect=RuntimeError("start failed")),
            self.assertRaisesRegex(RuntimeError, "start failed"),
        ):
            iterate_in_subprocess(_short_source)

        self.assertIsNone(process.pid)
        for ipc_queue in ipc_queues:
            with self.assertRaisesRegex(ValueError, "closed"):
                ipc_queue.put_nowait(None)
            _join_queue_thread(ipc_queue)

    def test_setup_cleanup_control_flow_exception_wins(self) -> None:
        """A cleanup SystemExit is not replaced by the setup failure."""
        process = MagicMock(pid=None)
        process.start.side_effect = RuntimeError("start failed")
        context, _ = _recording_context(
            MagicMock(),
            (MagicMock(), MagicMock()),
            process,
        )

        with (
            patch(
                "spdl.pipeline._iter_utils._subprocess.mp.get_context",
                return_value=context,
            ),
            patch.object(
                _ipc,
                "terminate",
                side_effect=SystemExit("cleanup interrupted"),
            ),
            self.assertRaisesRegex(SystemExit, "cleanup interrupted"),
        ):
            iterate_in_subprocess(_short_source)

    def test_initializer_timeout_reaps_partially_started_worker(self) -> None:
        """A subprocess that misses initialization timeout is reaped immediately."""
        context, ipc_queues, processes = self._make_real_context()
        release = mp.get_context("spawn").Event()

        try:
            with (
                patch(
                    "spdl.pipeline._iter_utils._subprocess.mp.get_context",
                    return_value=context,
                ),
                self.assertRaisesRegex(RuntimeError, "did not initialize"),
            ):
                iterate_in_subprocess(
                    _short_source,
                    initializer=partial(_blocked_initializer, release),
                    mp_context="spawn",
                    timeout=0.05,
                )
        finally:
            release.set()

        self.assertEqual(len(processes), 1)
        process = processes[0]
        self.addCleanup(_cleanup_resources, process, ipc_queues)
        self.assertFalse(process.is_alive())
        self.assertIsNotNone(process.exitcode)

    def test_dead_worker_is_detected_without_inactivity_timeout(self) -> None:
        """Unexpected subprocess death fails an unbounded result wait promptly."""
        release = mp.get_context("spawn").Event()
        iterable = iterate_in_subprocess(
            partial(_abrupt_process_exit_source, release),
            mp_context="spawn",
            timeout=None,
        )

        iterator = iter(iterable)
        self.assertEqual(next(iterator), 0)
        release.set()

        t0 = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "exited unexpectedly"):
            next(iterator)
        self.assertLess(time.monotonic() - t0, 2.0)

    def test_dead_worker_waits_for_delayed_terminal_message(self) -> None:
        """A delayed worker failure wins over the generic dead-worker error."""
        cmd_q = MagicMock()
        data_q = MagicMock()
        data_q.get.side_effect = [
            _Msg(_Status.INITIALIZATION_SUCCEEDED),
            _Msg(_Status.ITERATION_STARTED),
            queue.Empty,
            queue.Empty,
            queue.Empty,
            _Msg(_Status.ITERATOR_FAILED, "final worker failure"),
        ]
        data_q.get_nowait.side_effect = queue.Empty
        process = MagicMock()
        process.pid = 1
        process.exitcode = 1
        process.is_alive.return_value = False
        context = MagicMock()
        context.Queue.side_effect = [cmd_q, data_q]
        context.Process.return_value = process

        with patch(
            "spdl.pipeline._iter_utils._subprocess.mp.get_context",
            return_value=context,
        ):
            iterable = iterate_in_subprocess(_short_source)
            with self.assertRaisesRegex(RuntimeError, "final worker failure"):
                next(iter(iterable))

        self.assertEqual(data_q.get.call_count, 6)

    def test_dead_worker_checks_for_message_at_grace_deadline(self) -> None:
        """A message visible at the grace boundary wins over generic failure."""
        terminal = _Msg(_Status.ITERATOR_FAILED, "final worker failure")
        data_q = MagicMock()
        data_q.get.side_effect = queue.Empty
        data_q.get_nowait.return_value = terminal

        with patch(
            "spdl.pipeline._iter_utils._common.time.monotonic",
            side_effect=[0.0, 1.0],
        ):
            result = _get_worker_message(data_q, 0.0, lambda: False, "subprocess")

        self.assertIs(result, terminal)
        data_q.get_nowait.assert_called_once_with()


if sys.version_info >= (3, 14):

    class _AbruptSubinterpreterExitIterable:
        def __iter__(self) -> Iterator[int]:
            yield 0
            time.sleep(0.2)
            raise SystemExit(17)

    def _abrupt_subinterpreter_exit_source() -> Iterable[int]:
        return _AbruptSubinterpreterExitIterable()

    class IterateInSubinterpreterLifecycleTest(unittest.TestCase):
        def test_dead_worker_is_detected_without_inactivity_timeout(self) -> None:
            """Unexpected subinterpreter death fails an unbounded wait promptly."""
            iterable = iterate_in_subinterpreter(
                _abrupt_subinterpreter_exit_source,
                timeout=None,
            )

            iterator = iter(iterable)
            self.assertEqual(next(iterator), 0)

            t0 = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "exited unexpectedly"):
                next(iterator)
            self.assertLess(time.monotonic() - t0, 2.0)
