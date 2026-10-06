# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import threading
import unittest
from collections.abc import Iterable
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from spdl.pipeline import iterate_in_subprocess
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
