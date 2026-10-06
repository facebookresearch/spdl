# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import unittest
from collections.abc import Callable, Iterable, Iterator
from contextlib import suppress
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import call, MagicMock, patch

from spdl.pipeline import iterate_in_subinterpreter, iterate_in_subprocess
from spdl.pipeline._iter_utils._common import (
    _Cmd,
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
    import concurrent.interpreters

    from spdl.pipeline._iter_utils import _subinterpreter as _subinterpreter_impl

    def _briefly_blocked_initializer() -> None:
        time.sleep(0.2)

    def _failing_initializer() -> None:
        raise RuntimeError("initializer failed")

    class _CleanupFailure(BaseException):
        pass

    class _AbruptSubinterpreterExitIterable:
        def __iter__(self) -> Iterator[int]:
            yield 0
            time.sleep(0.2)
            raise SystemExit(17)

    def _abrupt_subinterpreter_exit_source() -> Iterable[int]:
        return _AbruptSubinterpreterExitIterable()

    class IterateInSubinterpreterLifecycleTest(unittest.TestCase):
        def test_completed_worker_cleanup_is_idempotent(self) -> None:
            """Repeated cleanup closes a completed worker only once."""
            thread = MagicMock()
            thread.is_alive.return_value = False
            interpreter = MagicMock()
            cmd_q = MagicMock()
            data_q = MagicMock()
            data_q.get_nowait.side_effect = queue.Empty
            interface = _subinterpreter_impl._iic(
                thread,
                interpreter,
                cmd_q,
                data_q,
                1.0,
            )

            interface.terminate()
            interface.terminate()

            thread.join.assert_called_once_with(timeout=3)
            interpreter.close.assert_called_once_with()
            cmd_q.put.assert_not_called()

        def test_failed_interpreter_close_can_be_retried(self) -> None:
            """A transient close failure retains the interpreter for a retry."""
            thread = MagicMock()
            thread.is_alive.return_value = False
            interpreter = MagicMock()
            interpreter.close.side_effect = [RuntimeError("close failed"), None]
            data_q = MagicMock()
            data_q.get_nowait.side_effect = queue.Empty
            interface = _subinterpreter_impl._iic(
                thread,
                interpreter,
                MagicMock(),
                data_q,
                1.0,
            )

            with self.assertLogs(
                "spdl.pipeline._iter_utils._subinterpreter", level="WARNING"
            ):
                interface.terminate()
            interface.terminate()

            self.assertEqual(interpreter.close.call_count, 2)
            self.assertIsNone(interface.interpreter)

        def test_stuck_worker_defers_interpreter_cleanup(self) -> None:
            """A live worker is closed once its deferred reaper observes exit."""
            thread = MagicMock()
            thread.daemon = False
            thread.is_alive.side_effect = [True, True, True, True, False]
            interpreter = MagicMock()
            cmd_q = MagicMock()
            data_q = MagicMock()
            data_q.get_nowait.side_effect = queue.Empty
            interface = _subinterpreter_impl._iic(
                thread,
                interpreter,
                cmd_q,
                data_q,
                1.0,
            )
            cleanup_thread = MagicMock()

            with (
                patch.object(
                    _subinterpreter_impl.threading,
                    "Thread",
                    return_value=cleanup_thread,
                ) as make_thread,
                self.assertLogs(
                    "spdl.pipeline._iter_utils._subinterpreter", level="DEBUG"
                ) as logs,
            ):
                interface.terminate()
                interface.terminate()

            cmd_q.put.assert_called_once_with(_Cmd.ABORT)
            cleanup_thread.start.assert_called_once_with()
            interpreter.close.assert_not_called()
            self.assertTrue(interface._cleanup_scheduled)
            self.assertTrue(make_thread.call_args.kwargs["daemon"])
            self.assertTrue(
                any(record.levelno == logging.DEBUG for record in logs.records)
            )

            cleanup_target = make_thread.call_args.kwargs["target"]
            with self.assertLogs(
                "spdl.pipeline._iter_utils._subinterpreter", level="WARNING"
            ) as deferred_logs:
                cleanup_target()

            interpreter.close.assert_called_once_with()
            self.assertEqual(
                thread.join.call_args_list,
                [
                    call(timeout=3),
                    call(timeout=60.0),
                    call(timeout=60.0),
                    call(timeout=60.0),
                ],
            )
            self.assertEqual(
                sum(
                    "deferred cleanup remains pending" in record.getMessage()
                    for record in deferred_logs.records
                ),
                1,
            )
            self.assertIsNone(interface.interpreter)
            self.assertFalse(interface._cleanup_scheduled)

        def test_deferred_close_base_exception_is_logged_and_retryable(self) -> None:
            """A reaper close BaseException stays visible and permits retry."""
            worker = MagicMock()
            worker.is_alive.return_value = False
            interpreter = MagicMock()
            interpreter.close.side_effect = [SystemExit("close failed"), None]
            data_q = MagicMock()
            data_q.get_nowait.side_effect = queue.Empty
            interface = _subinterpreter_impl._iic(
                worker,
                interpreter,
                MagicMock(),
                data_q,
                1.0,
            )
            interface._cleanup_scheduled = True

            with self.assertLogs(
                "spdl.pipeline._iter_utils._subinterpreter", level="WARNING"
            ) as logs:
                with self.assertRaisesRegex(SystemExit, "close failed"):
                    interface._close_after_worker_exit()

            self.assertTrue(
                any(
                    "during deferred cleanup" in record.getMessage()
                    for record in logs.records
                )
            )
            self.assertIs(interface.interpreter, interpreter)
            self.assertFalse(interface._cleanup_scheduled)

            interface.terminate()

            self.assertEqual(interpreter.close.call_count, 2)
            self.assertIsNone(interface.interpreter)

        def test_failed_deferred_join_allows_cleanup_retry(self) -> None:
            """A failed reaper join cannot permanently suppress terminate()."""
            worker = MagicMock()
            worker.is_alive.return_value = False
            worker.join.side_effect = [RuntimeError("join failed"), None]
            interpreter = MagicMock()
            data_q = MagicMock()
            data_q.get_nowait.side_effect = queue.Empty
            interface = _subinterpreter_impl._iic(
                worker,
                interpreter,
                MagicMock(),
                data_q,
                1.0,
            )
            interface._cleanup_scheduled = True

            with self.assertLogs(
                "spdl.pipeline._iter_utils._subinterpreter", level="WARNING"
            ):
                interface._close_after_worker_exit()

            self.assertFalse(interface._cleanup_scheduled)
            interpreter.close.assert_not_called()

            interface.terminate()

            self.assertEqual(
                worker.join.call_args_list,
                [call(timeout=60.0), call(timeout=3)],
            )
            interpreter.close.assert_called_once_with()
            self.assertIsNone(interface.interpreter)

        def test_failed_reaper_construction_allows_cleanup_retry(self) -> None:
            """A failed reaper constructor cannot suppress a later retry."""
            worker = MagicMock()
            worker.daemon = False
            worker.is_alive.return_value = True
            interpreter = MagicMock()
            data_q = MagicMock()
            data_q.get_nowait.side_effect = queue.Empty
            interface = _subinterpreter_impl._iic(
                worker,
                interpreter,
                MagicMock(),
                data_q,
                1.0,
            )
            cleanup_thread = MagicMock()

            with (
                patch.object(
                    _subinterpreter_impl.threading,
                    "Thread",
                    side_effect=[RuntimeError("constructor failed"), cleanup_thread],
                ) as make_thread,
                self.assertLogs(
                    "spdl.pipeline._iter_utils._subinterpreter", level="WARNING"
                ),
            ):
                interface.terminate()
                self.assertFalse(interface._cleanup_scheduled)
                interface.terminate()

            self.assertEqual(make_thread.call_count, 2)
            cleanup_thread.start.assert_called_once_with()
            self.assertTrue(interface._cleanup_scheduled)

            cleanup_target = make_thread.call_args.kwargs["target"]
            worker.is_alive.return_value = False
            cleanup_target()

            interpreter.close.assert_called_once_with()
            self.assertIsNone(interface.interpreter)
            self.assertFalse(interface._cleanup_scheduled)

        def test_reaper_start_base_exception_allows_cleanup_retry(self) -> None:
            """A BaseException from reaper start cannot suppress a later retry."""
            worker = MagicMock()
            worker.daemon = False
            worker.is_alive.return_value = True
            interpreter = MagicMock()
            data_q = MagicMock()
            data_q.get_nowait.side_effect = queue.Empty
            interface = _subinterpreter_impl._iic(
                worker,
                interpreter,
                MagicMock(),
                data_q,
                1.0,
            )
            failed_cleanup_thread = MagicMock()
            failed_cleanup_thread.start.side_effect = SystemExit("start failed")
            cleanup_thread = MagicMock()

            with (
                patch.object(
                    _subinterpreter_impl.threading,
                    "Thread",
                    side_effect=[failed_cleanup_thread, cleanup_thread],
                ) as make_thread,
                self.assertLogs(
                    "spdl.pipeline._iter_utils._subinterpreter", level="WARNING"
                ),
            ):
                with self.assertRaisesRegex(SystemExit, "start failed"):
                    interface.terminate()
                self.assertFalse(interface._cleanup_scheduled)
                interface.terminate()

            self.assertEqual(make_thread.call_count, 2)
            failed_cleanup_thread.start.assert_called_once_with()
            cleanup_thread.start.assert_called_once_with()
            self.assertTrue(interface._cleanup_scheduled)

            cleanup_target = make_thread.call_args.kwargs["target"]
            worker.is_alive.return_value = False
            cleanup_target()

            interpreter.close.assert_called_once_with()
            self.assertIsNone(interface.interpreter)
            self.assertFalse(interface._cleanup_scheduled)

        def test_concurrent_terminate_retries_failed_reaper_start(self) -> None:
            """A concurrent cleanup retries after deferred-reaper startup fails."""
            coordination_timeout = 5.0
            worker = MagicMock()
            worker.daemon = False
            worker.is_alive.return_value = True
            interpreter = MagicMock()
            data_q = MagicMock()
            data_q.get_nowait.side_effect = queue.Empty
            interface = _subinterpreter_impl._iic(
                worker,
                interpreter,
                MagicMock(),
                data_q,
                1.0,
            )
            start_entered = threading.Event()
            allow_start_failure = threading.Event()
            first_done = threading.Event()
            second_entered = threading.Event()
            second_done = threading.Event()
            errors: list[BaseException] = []
            failed_cleanup_thread = MagicMock()
            cleanup_thread = MagicMock()

            def fail_start() -> None:
                start_entered.set()
                if not allow_start_failure.wait(coordination_timeout):
                    raise RuntimeError("test did not release reaper startup")
                raise RuntimeError("reaper start failed")

            def run_terminate(
                entered: threading.Event | None,
                finished: threading.Event,
            ) -> None:
                if entered is not None:
                    entered.set()
                try:
                    interface.terminate()
                except BaseException as error:  # noqa: B036 - surface thread failures
                    errors.append(error)
                finally:
                    finished.set()

            failed_cleanup_thread.start.side_effect = fail_start
            first = threading.Thread(
                target=run_terminate,
                args=(None, first_done),
            )
            second = threading.Thread(
                target=run_terminate,
                args=(second_entered, second_done),
            )

            with (
                patch.object(
                    _subinterpreter_impl.threading,
                    "Thread",
                    side_effect=[failed_cleanup_thread, cleanup_thread],
                ) as make_thread,
                self.assertLogs(
                    "spdl.pipeline._iter_utils._subinterpreter", level="WARNING"
                ),
            ):
                try:
                    first.start()
                    self.assertTrue(start_entered.wait(coordination_timeout))
                    second.start()
                    self.assertTrue(second_entered.wait(coordination_timeout))
                    self.assertFalse(second_done.is_set())
                finally:
                    allow_start_failure.set()
                    if first.ident is not None:
                        first.join(timeout=coordination_timeout)
                    if second.ident is not None:
                        second.join(timeout=coordination_timeout)

            self.assertTrue(first_done.is_set())
            self.assertTrue(second_done.is_set())
            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(make_thread.call_count, 2)
            failed_cleanup_thread.start.assert_called_once_with()
            cleanup_thread.start.assert_called_once_with()
            self.assertTrue(interface._cleanup_scheduled)
            self.assertIsNone(interface._terminate_owner)

            cleanup_target = make_thread.call_args.kwargs["target"]
            worker.is_alive.return_value = False
            cleanup_target()

            interpreter.close.assert_called_once_with()
            self.assertIsNone(interface.interpreter)
            self.assertFalse(interface._cleanup_scheduled)

        def test_concurrent_terminate_returns_during_worker_join(self) -> None:
            """Concurrent cleanup returns promptly while another caller joins."""
            coordination_timeout = 5.0
            worker = MagicMock()
            worker.is_alive.side_effect = [True, False]
            interpreter = MagicMock()
            join_started = threading.Event()
            allow_join = threading.Event()
            second_done = threading.Event()
            errors: list[Exception] = []

            def join_worker(*, timeout: float) -> None:
                self.assertEqual(timeout, 3)
                join_started.set()
                if not allow_join.wait(coordination_timeout):
                    raise RuntimeError("test did not release worker join")

            def run(
                action: Callable[[], None], finished: threading.Event | None = None
            ) -> None:
                try:
                    action()
                except Exception as error:
                    errors.append(error)
                finally:
                    if finished is not None:
                        finished.set()

            worker.join.side_effect = join_worker
            cmd_q = MagicMock()
            data_q = MagicMock()
            data_q.get_nowait.side_effect = queue.Empty
            interface = _subinterpreter_impl._iic(
                worker,
                interpreter,
                cmd_q,
                data_q,
                1.0,
            )

            first = threading.Thread(target=run, args=(interface.terminate,))
            second = threading.Thread(
                target=run, args=(interface.terminate, second_done)
            )
            with self.assertLogs(
                "spdl.pipeline._iter_utils._subinterpreter", level="DEBUG"
            ) as logs:
                try:
                    first.start()
                    self.assertTrue(join_started.wait(coordination_timeout))
                    second.start()
                    self.assertTrue(second_done.wait(coordination_timeout))
                    self.assertTrue(first.is_alive())
                finally:
                    allow_join.set()
                    if first.ident is not None:
                        first.join(timeout=coordination_timeout)
                    if second.ident is not None:
                        second.join(timeout=coordination_timeout)

            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
            self.assertEqual(errors, [])
            self.assertTrue(
                any(record.levelno == logging.DEBUG for record in logs.records)
            )
            cmd_q.put.assert_called_once_with(_Cmd.ABORT)
            worker.join.assert_called_once_with(timeout=3)
            interpreter.close.assert_called_once_with()
            self.assertIsNone(interface.interpreter)

        def test_thread_start_failure_closes_new_interpreter(self) -> None:
            """A call-in-thread failure closes the interpreter created for it."""
            before = {
                interpreter.id for interpreter in concurrent.interpreters.list_all()
            }

            try:
                with (
                    patch.object(
                        threading.Thread,
                        "start",
                        side_effect=RuntimeError("thread start failed"),
                    ),
                    self.assertRaisesRegex(RuntimeError, "thread start failed"),
                ):
                    iterate_in_subinterpreter(_short_source)

                after = {
                    interpreter.id for interpreter in concurrent.interpreters.list_all()
                }
                self.assertEqual(after, before)
            finally:
                # Keep a regression from leaking an interpreter into later tests.
                for interpreter in concurrent.interpreters.list_all():
                    if interpreter.id not in before:
                        with suppress(Exception):
                            interpreter.close()

        def test_orphan_cleanup_preserves_call_in_thread_base_exception(self) -> None:
            """A cleanup BaseException cannot mask worker-creation failure."""
            interpreter = MagicMock()
            interpreter.call_in_thread.side_effect = KeyboardInterrupt(
                "worker creation failed"
            )
            interpreter.close.side_effect = _CleanupFailure("cleanup failed")

            with (
                patch.object(
                    _subinterpreter_impl.concurrent.interpreters,
                    "create_queue",
                    side_effect=[MagicMock(), MagicMock()],
                ),
                patch.object(
                    _subinterpreter_impl.concurrent.interpreters,
                    "create",
                    return_value=interpreter,
                ),
                self.assertLogs(
                    "spdl.pipeline._iter_utils._subinterpreter", level="WARNING"
                ),
                self.assertRaisesRegex(
                    KeyboardInterrupt, "worker creation failed"
                ) as error,
            ):
                iterate_in_subinterpreter(_short_source)

            self.assertEqual(
                error.exception.__notes__,
                [
                    "Closing the orphaned subinterpreter also failed: "
                    "_CleanupFailure: cleanup failed"
                ],
            )

        def test_initialization_cleanup_preserves_base_exception(self) -> None:
            """A cleanup BaseException cannot mask initialization failure."""
            interpreter = MagicMock()
            interpreter.call_in_thread.return_value = MagicMock()

            with (
                patch.object(
                    _subinterpreter_impl.concurrent.interpreters,
                    "create_queue",
                    side_effect=[MagicMock(), MagicMock()],
                ),
                patch.object(
                    _subinterpreter_impl.concurrent.interpreters,
                    "create",
                    return_value=interpreter,
                ),
                patch.object(
                    _subinterpreter_impl,
                    "_wait_for_init",
                    side_effect=KeyboardInterrupt("initialization failed"),
                ),
                patch.object(
                    _subinterpreter_impl._iic,
                    "terminate",
                    side_effect=_CleanupFailure("cleanup failed"),
                ),
                self.assertLogs(
                    "spdl.pipeline._iter_utils._subinterpreter", level="WARNING"
                ),
                self.assertRaisesRegex(
                    KeyboardInterrupt, "initialization failed"
                ) as error,
            ):
                iterate_in_subinterpreter(_short_source)

            self.assertEqual(
                error.exception.__notes__,
                [
                    "Cleaning up the failed subinterpreter initialization also "
                    "failed: _CleanupFailure: cleanup failed"
                ],
            )

        def test_initializer_failure_closes_interpreter(self) -> None:
            """An initializer failure closes its completed interpreter."""
            before = {
                interpreter.id for interpreter in concurrent.interpreters.list_all()
            }

            with self.assertRaisesRegex(RuntimeError, "initializer failed"):
                iterate_in_subinterpreter(
                    _short_source,
                    initializer=_failing_initializer,
                )

            after = {
                interpreter.id for interpreter in concurrent.interpreters.list_all()
            }
            self.assertEqual(after, before)

        def test_initializer_timeout_closes_partially_started_interpreter(self) -> None:
            """A timed-out initializer is closed after cooperative termination."""
            before = {
                interpreter.id for interpreter in concurrent.interpreters.list_all()
            }

            with self.assertRaisesRegex(RuntimeError, "did not initialize"):
                iterate_in_subinterpreter(
                    _short_source,
                    initializer=_briefly_blocked_initializer,
                    timeout=0.05,
                )

            after = {
                interpreter.id for interpreter in concurrent.interpreters.list_all()
            }
            self.assertEqual(after, before)

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
