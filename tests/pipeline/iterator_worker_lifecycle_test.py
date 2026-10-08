# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import queue
import threading
import unittest
from collections.abc import Iterable
from functools import partial
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

from spdl.pipeline import iterate_in_subprocess
from spdl.pipeline._iter_utils._subprocess import (
    _ipc,
    _iterate_results_until_closed,
)


def _short_source() -> Iterable[int]:
    return range(1)


def _failing_source() -> Iterable[int]:
    yield from ()
    raise RuntimeError("worker iteration failed")


def _blocked_initializer(release: Any) -> None:
    release.wait()


def _hold_queue_write_lock(ipc_queue: Any, ready: Any) -> None:
    ipc_queue._wlock.acquire()
    ready.set()
    threading.Event().wait()


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

        drain.assert_called_once_with(data_q)
        process.join.assert_called_once_with(3)
        arena.close.assert_called_once_with()
        arena.unlink.assert_called_once_with()
        for ipc_queue in (cmd_q, data_q):
            ipc_queue.cancel_join_thread.assert_called_once_with()
            ipc_queue.join_thread.assert_not_called()
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
        drain.assert_called_once_with(data_q)
        process.join.assert_called_once_with(3)
        arena.close.assert_called_once_with()
        arena.unlink.assert_called_once_with()
        for ipc_queue in (cmd_q, data_q):
            ipc_queue.cancel_join_thread.assert_called_once_with()
            ipc_queue.join_thread.assert_not_called()
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
            ipc_queue.join_thread.assert_not_called()
            ipc_queue.close.assert_called_once_with()

    def test_arena_close_failure_preserves_cleanup_order(self) -> None:
        """An arena close error cannot skip later IPC cleanup."""
        events: list[str] = []
        process = MagicMock(pid=1)
        process.is_alive.return_value = False
        cmd_q = MagicMock()
        data_q = MagicMock()
        arena = MagicMock()
        interface = _ipc(process, cmd_q, data_q, 1.0, arena)

        arena.shutdown_arena.side_effect = partial(events.append, "arena.shutdown")
        arena.unlink.side_effect = partial(events.append, "arena.unlink")
        cmd_q.cancel_join_thread.side_effect = partial(events.append, "cmd.cancel")
        cmd_q.close.side_effect = partial(events.append, "cmd.close")
        data_q.cancel_join_thread.side_effect = partial(events.append, "data.cancel")
        data_q.close.side_effect = partial(events.append, "data.close")

        def close_arena() -> None:
            events.append("arena.close")
            raise RuntimeError("arena close failed")

        arena.close.side_effect = close_arena

        with (
            patch(
                "spdl.pipeline._iter_utils._subprocess._drain",
                side_effect=lambda _data_q: events.append("drain"),
            ),
            patch(
                "spdl.pipeline._iter_utils._subprocess._join",
                side_effect=lambda _process: events.append("join"),
            ),
            self.assertRaisesRegex(RuntimeError, "arena close failed"),
        ):
            interface.terminate()

        self.assertEqual(
            events,
            [
                "arena.shutdown",
                "drain",
                "join",
                "arena.close",
                "arena.unlink",
                "cmd.cancel",
                "cmd.close",
                "data.cancel",
                "data.close",
            ],
        )
        self.assertIsNone(interface.cmd_q)
        self.assertIsNone(interface.data_q)

    def test_killed_result_writer_does_not_block_queue_cleanup(self) -> None:
        """A worker killed during a result write cannot wedge parent teardown."""
        ctx = mp.get_context("fork")
        cmd_q: Any = ctx.Queue()
        data_q: Any = ctx.Queue(maxsize=3)
        ready = ctx.Event()
        process = cast(
            mp.Process,
            ctx.Process(
                target=_hold_queue_write_lock,
                args=(data_q, ready),
            ),
        )
        ipc_queues = (cmd_q, data_q)
        self.addCleanup(_cleanup_resources, process, ipc_queues)

        process.start()
        self.assertTrue(ready.wait(timeout=5))
        process.kill()
        process.join(timeout=5)
        self.assertFalse(process.is_alive())

        interface = _ipc(process, cmd_q, data_q, 1.0)
        done = threading.Event()
        errors: list[Exception] = []

        def terminate() -> None:
            try:
                interface.terminate()
            except Exception as error:
                errors.append(error)
            finally:
                done.set()

        termination = threading.Thread(target=terminate, daemon=True)
        termination.start()
        returned_while_lock_poisoned = done.wait(timeout=2)

        try:
            data_q._wlock.release()
        except ValueError:
            pass
        termination.join(timeout=5)

        self.assertFalse(termination.is_alive())
        if errors:
            raise errors[0]
        self.assertTrue(
            returned_while_lock_poisoned,
            "queue cleanup waited for a write lock orphaned by the worker",
        )

    def test_result_queue_failure_propagates_before_shutdown(self) -> None:
        """Unexpected result queue failures remain visible before teardown."""
        data_q = MagicMock()
        interface = _ipc(MagicMock(), MagicMock(), data_q, 1.0)

        with (
            patch(
                "spdl.pipeline._iter_utils._subprocess._iterate_results",
                side_effect=OSError("broken result queue"),
            ),
            self.assertRaisesRegex(OSError, "broken result queue"),
        ):
            list(_iterate_results_until_closed(interface, data_q))

    def test_result_corruption_propagates_during_shutdown(self) -> None:
        """Shutdown cannot hide an in-flight result deserialization failure."""
        read_started = threading.Event()
        release_failure = threading.Event()
        errors: list[Exception] = []
        process = MagicMock(pid=1, exitcode=0)
        process.is_alive.return_value = False
        cmd_q = MagicMock()
        data_q = MagicMock()
        data_q.get_nowait.side_effect = queue.Empty
        interface = _ipc(process, cmd_q, data_q, 1.0)

        def fail_during_shutdown(*_args: Any, **_kwargs: Any) -> Any:
            read_started.set()
            if not release_failure.wait(timeout=10):
                raise RuntimeError("Timed out waiting for shutdown.")
            raise ValueError("corrupt result payload")

        def read_result() -> None:
            try:
                list(_iterate_results_until_closed(interface, data_q))
            except Exception as error:
                errors.append(error)

        data_q.get.side_effect = fail_during_shutdown
        reader = threading.Thread(target=read_result)
        reader.start()
        try:
            self.assertTrue(read_started.wait(timeout=10))
            interface.terminate()
        finally:
            release_failure.set()
            reader.join(timeout=10)

        self.assertFalse(reader.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ValueError)
        self.assertEqual(str(errors[0]), "corrupt result payload")

    def test_stale_discard_failure_does_not_skip_real_teardown(self) -> None:
        """A stale payload error still reaps a real worker and closes its queues."""
        for error in (
            RuntimeError("received 0 items of ancdata"),
            ValueError(),
        ):
            with self.subTest(error=type(error).__name__):
                source, process, ipc_queues = self._start_source_with_real_ipc()

                with (
                    patch(
                        "spdl.pipeline._iter_utils._subprocess._drain",
                        side_effect=error,
                    ),
                    self.assertRaisesRegex(RuntimeError, "worker iteration failed"),
                ):
                    list(source)

                self._assert_real_resources_released(process, ipc_queues)

    def test_closed_result_queue_during_discard_does_not_skip_teardown(self) -> None:
        """A concurrent queue close remains an expected teardown race."""
        process = MagicMock(pid=1, exitcode=0)
        process.is_alive.return_value = False
        cmd_q = MagicMock()
        data_q = MagicMock()
        interface = _ipc(process, cmd_q, data_q, 1.0)

        with (
            patch(
                "spdl.pipeline._iter_utils._subprocess._drain",
                side_effect=ValueError(f"Queue {data_q!r} is closed"),
            ),
            self.assertLogs("spdl.pipeline._iter_utils._subprocess", level="DEBUG"),
        ):
            interface.terminate()

        process.join.assert_called_once_with(3)
        self.assertIsNone(interface.cmd_q)
        self.assertIsNone(interface.data_q)
        for ipc_queue in (cmd_q, data_q):
            ipc_queue.cancel_join_thread.assert_called_once_with()
            ipc_queue.close.assert_called_once_with()

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
