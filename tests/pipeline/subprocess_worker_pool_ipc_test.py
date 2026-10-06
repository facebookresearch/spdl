# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for failures in the hoisted worker-pool IPC."""

import errno
import multiprocessing as mp
import os
import pickle
import queue
import threading
import unittest
import warnings
from concurrent.futures import BrokenExecutor, Future
from typing import Any
from unittest.mock import call, MagicMock, patch

from spdl.pipeline._subprocess_worker_pool import (
    _handle_queue_feeder_error,
    _OUTPUT_FEEDER_JOIN_TIMEOUT,
    _POOL_SHUTDOWN_REASON,
    _RemoteExecutor,
    _Result,
    _SHUTDOWN,
    _shutdown_pools,
    _start_pool_monitors,
    _WorkerPool,
    _WorkerStatus,
)

_PROCESS_READY_TIMEOUT: float = 30.0
_FUTURE_COMPLETION_TIMEOUT: float = 30.0


class _Unpicklable:
    """Payload whose reducer reproduces a queue-feeder serialization failure."""

    def __reduce__(self) -> Any:
        raise TypeError("cannot pickle test payload")


class _UnpicklableError(RuntimeError):
    """Worker exception that cannot itself cross the process boundary."""

    def __reduce__(self) -> Any:
        raise TypeError("cannot pickle test exception")


class _PicklingBaseException(BaseException):
    """A direct ``BaseException`` raised only by the test reducer below."""


class _UnpicklableWithBaseException:
    """Payload whose reducer bypasses ``multiprocessing.Queue``'s normal catch."""

    def __reduce__(self) -> Any:
        raise _PicklingBaseException("direct base exception while pickling")


def _identity(value: Any) -> Any:
    return value


def _return_unpicklable() -> Any:
    return _Unpicklable()


def _return_unpicklable_with_base_exception() -> Any:
    return _UnpicklableWithBaseException()


def _raise_unpicklable() -> None:
    raise _UnpicklableError("worker failed")


def _rebuild_serialization_probe(payload: bytes, calls: int) -> tuple[bytes, int]:
    return payload, calls


class _SerializationProbe:
    """Large payload that records where and how often its reducer runs."""

    def __init__(self, forbidden_thread: int, payload: bytes) -> None:
        self._forbidden_thread = forbidden_thread
        self._payload = payload
        self.calls = 0

    def __reduce__(self) -> Any:
        if threading.get_ident() == self._forbidden_thread:
            raise AssertionError("payload was serialized on its submitting thread")
        self.calls += 1
        return _rebuild_serialization_probe, (self._payload, self.calls)


def _return_serialization_probe(payload: bytes) -> Any:
    return _SerializationProbe(threading.get_ident(), payload)


def _block_initializer(started: Any) -> None:
    started.set()
    threading.Event().wait()


def _signal_initializer(started: Any) -> None:
    started.set()


def _exit_during_shutdown_initializer(started: Any, release: Any) -> None:
    started.set()
    release.wait()
    os._exit(17)


_EXIT_ENTERED: Any = None
_EXIT_RELEASE: Any = None


def _install_exit_gate(entered: Any, release: Any) -> None:
    global _EXIT_ENTERED, _EXIT_RELEASE
    _EXIT_ENTERED = entered
    _EXIT_RELEASE = release


def _exit_worker() -> None:
    _EXIT_ENTERED.set()
    _EXIT_RELEASE.wait()
    os._exit(23)


def _exercise_abrupt_worker_exit(executor: Any, submitted: Any, status: Any) -> None:
    """Submit crashing and pending work through a spawned remote process."""
    try:
        futures = [
            executor.submit(_exit_worker),
            executor.submit(_identity, 17),
        ]
        submitted.set()
        outcomes = []
        for future in futures:
            try:
                future.result(timeout=_FUTURE_COMPLETION_TIMEOUT)
            except Exception as error:
                outcomes.append((type(error).__name__, str(error)))
            else:
                outcomes.append(("result", ""))
        try:
            executor.submit(_identity, 1)
        except Exception as error:
            outcomes.append((type(error).__name__, str(error)))
        else:
            outcomes.append(("accepted", ""))
        status.send(outcomes)
    except Exception as error:
        status.send([("harness", f"{type(error).__name__}: {error}")])
    finally:
        status.close()


def _submit_then_exit(executor: Any, status: Any) -> None:
    """Use the remote executor, then exit with its liveness watcher still blocked."""
    try:
        status.send(
            executor.submit(_identity, 31).result(timeout=_FUTURE_COMPLETION_TIMEOUT)
        )
        # Give the daemon watcher time to enter its blocking receive before process teardown.
        # With a multiprocessing.Event, killing a registered waiter leaves Condition.notify
        # waiting forever for an acknowledgement that the now-dead process cannot provide.
        threading.Event().wait(0.2)
    except Exception as error:
        status.send(f"{type(error).__name__}: {error}")
    finally:
        status.close()


class _SerializationSignal:
    """Large payload that signals when the input feeder starts serializing it."""

    def __init__(self, serialized: threading.Event, payload: bytes) -> None:
        self._serialized = serialized
        self._payload = payload

    def __reduce__(self) -> Any:
        self._serialized.set()
        return bytes, (self._payload,)


class _GatedResultQueue:
    """Pause a result after dequeue so watcher/result ordering is deterministic."""

    def __init__(self, queue: Any) -> None:
        self._queue = queue
        self.result_dequeued = threading.Event()
        self.release_result = threading.Event()

    def get(self) -> Any:
        result = self._queue.get()
        self.result_dequeued.set()
        if not self.release_result.wait(timeout=_FUTURE_COMPLETION_TIMEOUT):
            raise TimeoutError("test did not release the dequeued result")
        return result


class _CorruptWorkerStatusChannel:
    """Status endpoint that reproduces a receive-side deserialization failure."""

    def recv(self) -> Any:
        raise pickle.UnpicklingError("corrupt worker status")

    def close(self) -> None:
        pass


class _FailureDuringJoinMonitor(threading.Thread):
    """Publish a late worker failure while shutdown joins the monitor."""

    def __init__(self, worker_failed: threading.Event) -> None:
        super().__init__(daemon=True)
        self._worker_failed = worker_failed
        self.join_timeout: float | None = -1

    def join(self, timeout: float | None = None) -> None:
        self.join_timeout = timeout
        self._worker_failed.set()


class _StuckMonitor(threading.Thread):
    """Model a monitor whose worker sentinel never becomes ready."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.join_timeout: float | None = -1

    def join(self, timeout: float | None = None) -> None:
        self.join_timeout = timeout

    def is_alive(self) -> bool:
        return True


class WorkerPoolSerializationTest(unittest.TestCase):
    def test_input_transport_failure_fails_all_pending_futures(self) -> None:
        """A partial input frame is never replayed or manually re-accounted."""
        errors = (
            InterruptedError(errno.EINTR, "input pipe was interrupted"),
            BrokenPipeError(errno.EPIPE, "input pipe is broken"),
        )
        for error in errors:
            with self.subTest(error=error):
                in_q = MagicMock()
                executor = _RemoteExecutor(in_q, MagicMock(), 1, MagicMock())
                futures = [Future(), Future()]
                executor._futures.update({3: futures[0], 4: futures[1]})

                with patch(
                    "spdl.pipeline._subprocess_worker_pool.traceback.print_exception"
                ) as print_exception:
                    in_q._on_queue_feeder_error(
                        error, memoryview(b"partial serialized submission")
                    )

                in_q._writer.send_bytes.assert_not_called()
                in_q._sem.acquire.assert_not_called()
                in_q._sem.release.assert_not_called()
                in_q._wlock.acquire.assert_not_called()
                in_q._wlock.release.assert_not_called()
                print_exception.assert_called_once_with(
                    type(error), error, error.__traceback__
                )
                self.assertEqual(executor._futures, {})
                self.assertEqual(
                    executor._broken,
                    "Worker pool input queue transport failed after serialization.",
                )
                for future in futures:
                    with self.assertRaisesRegex(
                        BrokenExecutor, "input queue transport failed"
                    ):
                        future.result()

    def test_full_output_queue_retries_router_sentinel_with_bound(self) -> None:
        """A transient full queue gets one bounded router-sentinel retry."""
        for retry_error, expected_abandon, expected_status_reason in (
            (None, False, None),
            (queue.Full(), True, _POOL_SHUTDOWN_REASON),
        ):
            with self.subTest(retry_error=retry_error):
                pool = object.__new__(_WorkerPool)
                pool._closed = False
                pool._procs = []
                pool._shutdown_started = threading.Event()
                pool._monitor = None
                pool._worker_failed = threading.Event()
                pool._status_lock = threading.Lock()
                pool._status_sent = False
                pool._status_receiver_transferred = False
                pool._status_receiver_watcher_started = threading.Event()
                pool._worker_status_recv = MagicMock()
                pool._worker_status_send = MagicMock()
                pool._in_q = MagicMock()
                pool._out_q = MagicMock()
                in_q = pool._in_q
                out_q = pool._out_q
                out_q.put_nowait.side_effect = queue.Full
                out_q.put.side_effect = retry_error

                with patch.object(
                    _WorkerPool,
                    "_close_queue",
                    return_value=True,
                ) as close_queue:
                    pool.shutdown()

                out_q.put_nowait.assert_called_once_with(_SHUTDOWN)
                out_q.put.assert_called_once_with(
                    _SHUTDOWN,
                    timeout=_OUTPUT_FEEDER_JOIN_TIMEOUT,
                )
                self.assertEqual(
                    close_queue.call_args_list,
                    [
                        call(in_q, abandon=False),
                        call(
                            out_q,
                            abandon=expected_abandon,
                            feeder_join_timeout=_OUTPUT_FEEDER_JOIN_TIMEOUT,
                        ),
                    ],
                )
                pool._worker_status_send.send.assert_called_once_with(
                    _WorkerStatus(expected_status_reason)
                )

    def test_queue_close_error_does_not_skip_remaining_cleanup(self) -> None:
        """One broken queue handle cannot skip cleanup of the other queue."""
        pool = object.__new__(_WorkerPool)
        pool._closed = False
        pool._procs = []
        pool._shutdown_started = threading.Event()
        pool._monitor = None
        pool._worker_failed = threading.Event()
        pool._status_lock = threading.Lock()
        pool._status_sent = False
        pool._status_receiver_transferred = False
        pool._status_receiver_watcher_started = threading.Event()
        pool._worker_status_recv = MagicMock()
        pool._worker_status_send = MagicMock()
        in_q = pool._in_q = MagicMock(
            spec=["cancel_join_thread", "close", "join_thread", "put"]
        )
        out_q = pool._out_q = MagicMock(
            spec=["cancel_join_thread", "close", "join_thread", "put_nowait"]
        )
        in_q.close.side_effect = OSError("input queue is already closed")
        in_q.join_thread.side_effect = ValueError("input feeder is invalid")

        with self.assertLogs("spdl.pipeline._subprocess_worker_pool", level="WARNING"):
            pool.shutdown()

        in_q.close.assert_called_once_with()
        in_q.join_thread.assert_not_called()
        out_q.close.assert_called_once_with()
        out_q.cancel_join_thread.assert_called_once_with()
        out_q.join_thread.assert_not_called()

    def test_close_queue_without_private_feeder_skips_unbounded_join(self) -> None:
        """A compatible queue need not expose CPython's private _thread."""
        queue_without_thread = MagicMock(
            spec=["cancel_join_thread", "close", "join_thread"]
        )

        self.assertFalse(
            _WorkerPool._close_queue(
                queue_without_thread,
                abandon=False,
                feeder_join_timeout=0.01,
            )
        )

        queue_without_thread.close.assert_called_once_with()
        queue_without_thread.cancel_join_thread.assert_called_once_with()
        queue_without_thread.join_thread.assert_not_called()

    def test_close_queue_with_non_thread_private_feeder_skips_unbounded_join(
        self,
    ) -> None:
        """A compatible queue may expose unrelated private feeder state."""
        queue_with_other_thread_state = MagicMock(
            spec=["_thread", "cancel_join_thread", "close", "join_thread"]
        )
        queue_with_other_thread_state._thread = object()

        self.assertFalse(
            _WorkerPool._close_queue(
                queue_with_other_thread_state,
                abandon=False,
                feeder_join_timeout=0.01,
            )
        )

        queue_with_other_thread_state.close.assert_called_once_with()
        queue_with_other_thread_state.cancel_join_thread.assert_called_once_with()
        queue_with_other_thread_state.join_thread.assert_not_called()

    def test_stalled_feeder_skips_unbounded_public_join(self) -> None:
        """A failed cancellation cannot fall through to blocking join_thread()."""

        class _StalledFeeder(threading.Thread):
            def __init__(self) -> None:
                super().__init__(daemon=True)
                self.join_timeout: float | None = None

            def join(self, timeout: float | None = None) -> None:
                self.join_timeout = timeout

            def is_alive(self) -> bool:
                return True

        stalled_feeder = _StalledFeeder()
        worker_queue = MagicMock(
            spec=["_thread", "cancel_join_thread", "close", "join_thread"]
        )
        worker_queue._thread = stalled_feeder
        worker_queue.cancel_join_thread.side_effect = OSError("cancel failed")

        with self.assertLogs("spdl.pipeline._subprocess_worker_pool", level="WARNING"):
            feeder_stopped = _WorkerPool._close_queue(
                worker_queue,
                abandon=False,
                feeder_join_timeout=0.01,
            )

        self.assertFalse(feeder_stopped)
        self.assertEqual(stalled_feeder.join_timeout, 0.01)
        worker_queue.close.assert_called_once_with()
        worker_queue.cancel_join_thread.assert_called_once_with()
        worker_queue.join_thread.assert_not_called()

    def test_fail_pending_contains_callback_base_exception(self) -> None:
        """A hostile callback cannot stop failure fanout to later futures."""
        executor = _RemoteExecutor(MagicMock(), MagicMock(), 1, MagicMock())
        cancelled: Future[Any] = Future()
        callback_future: Future[Any] = Future()
        trailing: Future[Any] = Future()
        self.assertTrue(cancelled.cancel())

        def fail_callback(completed: Future[Any]) -> None:
            self.assertIs(completed, callback_future)
            raise _PicklingBaseException("callback failed")

        callback_future.add_done_callback(fail_callback)
        executor._futures[0] = cancelled
        executor._futures[1] = callback_future
        executor._futures[2] = trailing

        with self.assertLogs("spdl.pipeline._subprocess_worker_pool", level="WARNING"):
            executor._fail_pending("worker failed")

        self.assertTrue(cancelled.cancelled())
        self.assertIsInstance(callback_future.exception(), BrokenExecutor)
        with self.assertRaisesRegex(BrokenExecutor, "worker failed"):
            trailing.result()
        self.assertEqual(executor._futures, {})
        self.assertEqual(executor._broken, "worker failed")

    def test_fail_pending_defers_control_flow_error_until_after_fanout(self) -> None:
        """Process control flow propagates only after every future is failed."""
        for error in (KeyboardInterrupt("interrupt"), SystemExit(23)):
            with self.subTest(error=type(error).__name__):
                executor = _RemoteExecutor(MagicMock(), MagicMock(), 1, MagicMock())
                callback_future: Future[Any] = Future()
                trailing: Future[Any] = Future()

                def fail_callback(
                    completed: Future[Any],
                    expected: Future[Any] = callback_future,
                    raised: BaseException = error,
                ) -> None:
                    self.assertIs(completed, expected)
                    raise raised

                callback_future.add_done_callback(fail_callback)
                executor._futures[0] = callback_future
                executor._futures[1] = trailing

                with self.assertRaises(type(error)) as raised:
                    executor._fail_pending("worker failed")

                self.assertIs(raised.exception, error)
                self.assertIsInstance(callback_future.exception(), BrokenExecutor)
                with self.assertRaisesRegex(BrokenExecutor, "worker failed"):
                    trailing.result()
                self.assertEqual(executor._futures, {})

    def test_preexisting_broken_executor_closes_worker_status(self) -> None:
        """Repeated broken submissions close an invalid status endpoint only once."""
        worker_status = MagicMock()
        worker_status.close.side_effect = OSError("status endpoint already closed")
        executor = _RemoteExecutor(MagicMock(), MagicMock(), 1, worker_status)
        executor._broken = "helper startup failed"

        with self.assertLogs(
            "spdl.pipeline._subprocess_worker_pool", level="ERROR"
        ) as logs:
            for value in (1, 2):
                with self.assertRaisesRegex(BrokenExecutor, "helper startup failed"):
                    executor.submit(_identity, value)

        worker_status.close.assert_called_once_with()
        self.assertEqual(len(logs.records), 1)
        self.assertIsNone(executor._thread)
        self.assertIsNone(executor._worker_watcher)

    def test_submission_failure_contains_callback_base_exception(self) -> None:
        """A failing done callback cannot escape the queue-feeder error path."""
        executor = _RemoteExecutor(MagicMock(), MagicMock(), 1, MagicMock())
        future: Future[Any] = Future()
        task_id = 7
        executor._futures[task_id] = future

        def fail_callback(completed: Future[Any]) -> None:
            self.assertIs(completed, future)
            raise _PicklingBaseException("callback failed")

        future.add_done_callback(fail_callback)
        error = TypeError("cannot pickle submission")
        with patch(
            "spdl.pipeline._subprocess_worker_pool.traceback.print_exc"
        ) as print_exc:
            executor._fail_submission(task_id, error)

        print_exc.assert_called_once_with()
        self.assertNotIn(task_id, executor._futures)
        self.assertIs(future.exception(), error)

    def test_failed_result_fallback_exits_worker(self) -> None:
        """A broken fallback transport makes the worker failure observable."""
        for error in (
            BrokenPipeError(errno.EPIPE, "result pipe is broken"),
            ConnectionResetError(errno.ECONNRESET, "result pipe was reset"),
        ):
            with self.subTest(error=error):
                out_q = MagicMock()
                out_q._closed = False
                out_q.put.side_effect = error
                result = _Result(out_q, 0, True, object())

                with (
                    patch(
                        "spdl.pipeline._subprocess_worker_pool.traceback.print_exc"
                    ) as print_exc,
                    patch(
                        "spdl.pipeline._subprocess_worker_pool.os._exit"
                    ) as exit_worker,
                ):
                    result._on_queue_feeder_error(TypeError("cannot pickle result"))

                out_q.put.assert_called_once()
                print_exc.assert_called_once_with()
                exit_worker.assert_called_once_with(1)

    def test_failed_result_fallback_retries_transient_transport_errors(self) -> None:
        """Repeated transient transport interruptions preserve the fallback."""
        out_q = MagicMock()
        out_q._closed = False
        out_q.put.side_effect = [
            InterruptedError(errno.EINTR, "interrupted"),
            BlockingIOError(errno.EAGAIN, "try again"),
            BlockingIOError(errno.EWOULDBLOCK, "would block"),
            None,
        ]
        result = _Result(out_q, 0, True, object())

        with (
            patch(
                "spdl.pipeline._subprocess_worker_pool.traceback.print_exc"
            ) as print_exc,
            patch("spdl.pipeline._subprocess_worker_pool.os._exit") as exit_worker,
            patch("spdl.pipeline._subprocess_worker_pool.time.sleep") as sleep,
        ):
            result._on_queue_feeder_error(TypeError("cannot pickle result"))

        self.assertEqual(out_q.put.call_count, 4)
        self.assertEqual(
            [record.args for record in sleep.call_args_list],
            [(0.001,), (0.002,), (0.004,)],
        )
        print_exc.assert_not_called()
        exit_worker.assert_not_called()

    def test_failed_result_fallback_exits_after_transient_transport_budget(
        self,
    ) -> None:
        """Persistent transient transport errors eventually terminate the worker."""
        out_q = MagicMock()
        out_q._closed = False
        out_q.put.side_effect = InterruptedError(errno.EINTR, "interrupted")
        result = _Result(out_q, 0, True, object())

        with (
            patch(
                "spdl.pipeline._subprocess_worker_pool.traceback.print_exc"
            ) as print_exc,
            patch("spdl.pipeline._subprocess_worker_pool.os._exit") as exit_worker,
            patch("spdl.pipeline._subprocess_worker_pool.time.sleep") as sleep,
        ):
            result._on_queue_feeder_error(TypeError("cannot pickle result"))

        self.assertEqual(out_q.put.call_count, 8)
        self.assertEqual(
            [record.args for record in sleep.call_args_list],
            [(0.001,), (0.002,), (0.004,), (0.008,), (0.016,), (0.032,), (0.05,)],
        )
        print_exc.assert_called_once_with()
        exit_worker.assert_called_once_with(1)

    def test_failed_result_fallback_retries_full_queue(self) -> None:
        """A one-off full result queue does not terminate the worker."""
        out_q = MagicMock()
        out_q._closed = False
        out_q.put.side_effect = [queue.Full(), None]
        result = _Result(out_q, 0, True, object())

        with (
            patch(
                "spdl.pipeline._subprocess_worker_pool.traceback.print_exc"
            ) as print_exc,
            patch("spdl.pipeline._subprocess_worker_pool.os._exit") as exit_worker,
            patch("spdl.pipeline._subprocess_worker_pool.time.sleep") as sleep,
        ):
            result._on_queue_feeder_error(TypeError("cannot pickle result"))

        self.assertEqual(out_q.put.call_count, 2)
        self.assertEqual(
            [record.kwargs for record in out_q.put.call_args_list],
            [{"block": False}, {"block": False}],
        )
        sleep.assert_called_once_with(0.001)
        print_exc.assert_not_called()
        exit_worker.assert_not_called()

    def test_result_transport_failure_exits_worker(self) -> None:
        """A dropped serialized result terminates its worker instead of hanging a Future."""
        error = BrokenPipeError(errno.EPIPE, "result pipe is broken")

        with (
            patch(
                "spdl.pipeline._subprocess_worker_pool.traceback.print_exception"
            ) as print_exception,
            patch("spdl.pipeline._subprocess_worker_pool.os._exit") as exit_worker,
        ):
            _handle_queue_feeder_error(
                error, memoryview(b"serialized result with a partial frame")
            )

        print_exception.assert_called_once_with(type(error), error, error.__traceback__)
        exit_worker.assert_called_once_with(1)

    def test_failed_result_fallback_caps_full_queue_backoff(self) -> None:
        """Repeated full-queue backoff is capped while retry budget remains."""
        out_q = MagicMock()
        out_q._closed = False
        out_q.put.side_effect = [queue.Full() for _ in range(7)] + [None]
        result = _Result(out_q, 0, True, object())

        with (
            patch(
                "spdl.pipeline._subprocess_worker_pool.traceback.print_exc"
            ) as print_exc,
            patch("spdl.pipeline._subprocess_worker_pool.os._exit") as exit_worker,
            patch("spdl.pipeline._subprocess_worker_pool.time.sleep") as sleep,
        ):
            result._on_queue_feeder_error(TypeError("cannot pickle result"))

        self.assertEqual(out_q.put.call_count, 8)
        self.assertEqual(
            [record.args for record in sleep.call_args_list],
            [(0.001,), (0.002,), (0.004,), (0.008,), (0.016,), (0.032,), (0.05,)],
        )
        print_exc.assert_not_called()
        exit_worker.assert_not_called()

    def test_failed_result_fallback_exits_after_full_queue_budget(self) -> None:
        """Persistent result backpressure eventually makes worker failure observable."""
        out_q = MagicMock()
        out_q._closed = False
        out_q.put.side_effect = queue.Full()
        result = _Result(out_q, 0, True, object())

        with (
            patch(
                "spdl.pipeline._subprocess_worker_pool.traceback.print_exc"
            ) as print_exc,
            patch("spdl.pipeline._subprocess_worker_pool.os._exit") as exit_worker,
            patch("spdl.pipeline._subprocess_worker_pool.time.sleep") as sleep,
        ):
            result._on_queue_feeder_error(TypeError("cannot pickle result"))

        self.assertEqual(out_q.put.call_count, 8)
        self.assertEqual(
            [record.args for record in sleep.call_args_list],
            [
                (0.001,),
                (0.002,),
                (0.004,),
                (0.008,),
                (0.016,),
                (0.032,),
                (0.05,),
            ],
        )
        print_exc.assert_called_once_with()
        exit_worker.assert_called_once_with(1)

    def test_failed_result_fallback_ignores_closed_queue(self) -> None:
        """A result fallback racing deliberate queue shutdown exits quietly."""
        out_q = MagicMock()
        out_q._closed = False

        def close_queue(*args: Any, **kwargs: Any) -> None:
            out_q._closed = True
            raise ValueError("Queue is closed")

        out_q.put.side_effect = close_queue
        result = _Result(out_q, 0, True, object())

        with (
            patch(
                "spdl.pipeline._subprocess_worker_pool.traceback.print_exc"
            ) as print_exc,
            patch("spdl.pipeline._subprocess_worker_pool.os._exit") as exit_worker,
        ):
            result._on_queue_feeder_error(TypeError("cannot pickle result"))

        out_q.put.assert_called_once()
        print_exc.assert_not_called()
        exit_worker.assert_not_called()

    def test_failed_result_fallback_skips_already_closed_queue(self) -> None:
        """A result fallback does not enqueue after deliberate queue shutdown."""
        out_q = MagicMock()
        out_q._closed = True
        result = _Result(out_q, 0, True, object())

        with (
            patch(
                "spdl.pipeline._subprocess_worker_pool.traceback.print_exc"
            ) as print_exc,
            patch("spdl.pipeline._subprocess_worker_pool.os._exit") as exit_worker,
        ):
            result._on_queue_feeder_error(TypeError("cannot pickle result"))

        out_q.put.assert_not_called()
        print_exc.assert_not_called()
        exit_worker.assert_not_called()

    def test_router_start_failure_breaks_executor_and_closes_status(self) -> None:
        """A failed router start cannot leave later submissions without a consumer."""
        pool = _WorkerPool(mp.get_context("spawn"), 1, None, ())
        executor = pool.make_executor()
        real_start = threading.Thread.start

        def _start(thread: threading.Thread) -> None:
            if thread.name == "spdl_remote_executor_router":
                raise RuntimeError("router start failed")
            real_start(thread)

        try:
            with patch.object(threading.Thread, "start", _start):
                with self.assertRaisesRegex(RuntimeError, "router start failed"):
                    executor.submit(_identity, 1)
            with self.assertRaisesRegex(BrokenExecutor, "router start failed"):
                executor.submit(_identity, 2)
            self.assertIsNone(executor._thread)
            self.assertIsNone(executor._worker_watcher)
            self.assertTrue(executor._worker_status.closed)
        finally:
            _shutdown_pools([pool])

    def test_watcher_start_failure_breaks_executor_and_stops_router(self) -> None:
        """A partial helper startup fails pending work and stops its live router."""
        pool = _WorkerPool(mp.get_context("spawn"), 1, None, ())
        executor = pool.make_executor()
        pending: Future[Any] = Future()
        executor._futures[7] = pending
        real_start = threading.Thread.start
        started_router: threading.Thread | None = None

        def _start(thread: threading.Thread) -> None:
            nonlocal started_router
            if thread.name == "spdl_remote_executor_worker_watcher":
                raise RuntimeError("watcher start failed")
            if thread.name == "spdl_remote_executor_router":
                started_router = thread
            real_start(thread)

        try:
            with patch.object(threading.Thread, "start", _start):
                with self.assertRaisesRegex(RuntimeError, "watcher start failed"):
                    executor.submit(_identity, 1)
            with self.assertRaisesRegex(BrokenExecutor, "watcher start failed"):
                executor.submit(_identity, 2)
            with self.assertRaisesRegex(BrokenExecutor, "watcher start failed"):
                pending.result(timeout=0)
            self.assertEqual(executor._futures, {})
            self.assertIsNone(executor._thread)
            self.assertIsNone(executor._worker_watcher)
            self.assertIsNotNone(started_router)
            self.assertTrue(started_router and not started_router.is_alive())
            self.assertTrue(executor._worker_status.closed)
            self.assertFalse(pool._status_receiver_watcher_started.is_set())
        finally:
            _shutdown_pools([pool])

    def test_status_pipe_registration_failure_closes_both_endpoints(self) -> None:
        """A failed after-fork registration cannot leak status-pipe descriptors."""
        ctx = MagicMock()
        in_q = MagicMock()
        out_q = MagicMock()
        proc = MagicMock()
        receive_status = MagicMock()
        send_status = MagicMock()
        ctx.Queue.side_effect = [in_q, out_q]
        ctx.Process.return_value = proc
        ctx.Pipe.return_value = (receive_status, send_status)

        with (
            patch(
                "spdl.pipeline._subprocess_worker_pool.register_after_fork",
                side_effect=RuntimeError("registration failed"),
            ),
            patch.object(_WorkerPool, "_terminate") as terminate,
        ):
            with self.assertRaisesRegex(RuntimeError, "registration failed"):
                _WorkerPool(ctx, 1, None, ())

        receive_status.close.assert_called_once_with()
        send_status.close.assert_called_once_with()
        terminate.assert_called_once_with([proc])
        for worker_queue in (in_q, out_q):
            worker_queue.close.assert_called_once_with()
            worker_queue.join_thread.assert_called_once_with()

    def test_worker_status_deserialization_failure_breaks_pending_work(self) -> None:
        """A corrupt status message cannot silently kill the sole watcher."""
        executor = _RemoteExecutor(
            MagicMock(), MagicMock(), 1, _CorruptWorkerStatusChannel()
        )
        future: Future[Any] = Future()
        executor._futures[0] = future

        with self.assertLogs("spdl.pipeline._subprocess_worker_pool", level="ERROR"):
            executor._watch_worker_pool()

        with self.assertRaisesRegex(
            BrokenExecutor,
            "Worker pool status watcher failed: UnpicklingError: corrupt worker status",
        ):
            future.result(timeout=0)
        self.assertEqual(
            executor._broken,
            "Worker pool status watcher failed: UnpicklingError: corrupt worker status",
        )

    def test_status_close_failure_still_breaks_pending_work(self) -> None:
        """A status-endpoint close failure cannot strand pending Futures."""
        status = MagicMock()
        status.recv.return_value = _WorkerStatus("worker failed")
        status.close.side_effect = OSError("status handle is closed")
        executor = _RemoteExecutor(MagicMock(), MagicMock(), 1, status)
        future: Future[Any] = Future()
        executor._futures[0] = future

        with self.assertLogs("spdl.pipeline._subprocess_worker_pool", level="ERROR"):
            executor._watch_worker_pool()

        with self.assertRaisesRegex(BrokenExecutor, "worker failed"):
            future.result(timeout=0)
        self.assertEqual(executor._broken, "worker failed")

    def test_failed_status_send_wakes_watcher_and_breaks_pending_work(self) -> None:
        """A failed status send closes its pipe so the watcher cannot hang."""
        receive_status, send_status = mp.get_context("spawn").Pipe(duplex=False)
        pool = object.__new__(_WorkerPool)
        pool._status_lock = threading.Lock()
        pool._status_sent = False
        sender = MagicMock()
        sender.send.side_effect = OSError("status send failed")
        sender.close.side_effect = send_status.close
        pool._worker_status_send = sender

        executor = _RemoteExecutor(MagicMock(), MagicMock(), 1, receive_status)
        future: Future[Any] = Future()
        executor._futures[0] = future
        watcher = threading.Thread(target=executor._watch_worker_pool, daemon=True)
        watcher.start()
        try:
            pool._send_worker_status("worker failed")
            pool._send_worker_status("later failure")
            with self.assertRaisesRegex(BrokenExecutor, "channel closed unexpectedly"):
                future.result(timeout=_FUTURE_COMPLETION_TIMEOUT)
            watcher.join(timeout=5)
            self.assertFalse(watcher.is_alive())
            self.assertTrue(pool._status_sent)
            sender.send.assert_called_once_with(_WorkerStatus("worker failed"))
            sender.close.assert_called_once_with()
        finally:
            send_status.close()
            receive_status.close()

    def test_same_process_status_receiver_closes_under_owner_lock(self) -> None:
        """A shared receive handle cannot close concurrently with owner status I/O."""
        pool = object.__new__(_WorkerPool)
        pool._in_q = MagicMock()
        pool._out_q = MagicMock()
        pool._max_workers = 1
        pool._status_lock = threading.Lock()
        pool._status_receiver_transferred = False
        pool._status_receiver_watcher_started = threading.Event()
        status = MagicMock()
        status.recv.return_value = _WorkerStatus(None)

        def _close() -> None:
            acquired = pool._status_lock.acquire(blocking=False)
            if acquired:
                pool._status_lock.release()
            self.assertFalse(acquired, "receive endpoint closed outside owner lock")

        status.close.side_effect = _close
        pool._worker_status_recv = status
        executor = pool.make_executor()

        executor._watch_worker_pool()

        status.close.assert_called_once_with()

    def test_executor_construction_failure_keeps_receiver_owner_cleanup(self) -> None:
        """A failed executor handoff leaves the receive endpoint with the pool."""
        pool = object.__new__(_WorkerPool)
        pool._closed = False
        pool._procs = []
        pool._shutdown_started = threading.Event()
        pool._monitor = None
        pool._worker_failed = threading.Event()
        pool._max_workers = 1
        pool._status_lock = threading.Lock()
        pool._status_sent = False
        pool._status_receiver_transferred = False
        pool._status_receiver_watcher_started = threading.Event()
        pool._worker_status_recv = MagicMock()
        pool._worker_status_send = MagicMock()
        pool._in_q = MagicMock(
            spec=["cancel_join_thread", "close", "join_thread", "put"]
        )
        pool._out_q = MagicMock(
            spec=["cancel_join_thread", "close", "join_thread", "put_nowait"]
        )

        def fail_construction(*args: Any, **kwargs: Any) -> None:
            acquired = pool._status_lock.acquire(blocking=False)
            if acquired:
                pool._status_lock.release()
            self.assertFalse(acquired, "executor constructed outside owner lock")
            raise RuntimeError("executor construction failed")

        with (
            patch(
                "spdl.pipeline._subprocess_worker_pool._RemoteExecutor",
                side_effect=fail_construction,
            ),
            self.assertRaisesRegex(RuntimeError, "executor construction failed"),
        ):
            pool.make_executor()

        self.assertFalse(pool._status_receiver_transferred)
        pool.shutdown()
        pool._worker_status_recv.close.assert_called_once_with()

    def test_unused_same_process_executor_receiver_closes_on_shutdown(self) -> None:
        """Pool shutdown reclaims a shared receiver if its watcher never starts."""
        pool = _WorkerPool(mp.get_context("spawn"), 1, None, ())
        executor = pool.make_executor()
        try:
            self.assertTrue(pool._status_receiver_transferred)
            self.assertFalse(pool._status_receiver_watcher_started.is_set())
            self.assertIsNone(executor._worker_watcher)
        finally:
            pool.shutdown()

        self.assertTrue(pool._worker_status_recv.closed)
        executor._watch_worker_pool()
        self.assertEqual(
            executor._broken,
            "The worker pool status channel closed unexpectedly.",
        )
        with self.assertRaisesRegex(
            BrokenExecutor,
            "status channel closed unexpectedly",
        ):
            executor.submit(_identity, 1)

    def test_owner_status_sender_closes_under_status_lock(self) -> None:
        """Final sender close cannot race a monitor's serialized status send."""
        pool = _WorkerPool(mp.get_context("spawn"), 1, None, ())
        real_sender = pool._worker_status_send
        sender = MagicMock()
        sender.send.side_effect = real_sender.send

        def _close() -> None:
            acquired = pool._status_lock.acquire(blocking=False)
            if acquired:
                pool._status_lock.release()
            self.assertFalse(acquired, "send endpoint closed outside status lock")
            real_sender.close()

        sender.close.side_effect = _close
        pool._worker_status_send = sender
        try:
            pool.shutdown()
        finally:
            real_sender.close()

        sender.close.assert_called_once_with()

    def test_clean_worker_exit_keeps_graceful_queue_cleanup(self) -> None:
        """A clean exit flushes both queue feeders before closing them."""
        ctx = mp.get_context("spawn")
        pool = _WorkerPool(ctx, 1, None, ())
        in_q = pool._in_q
        out_q = pool._out_q
        try:
            executor = pool.make_executor()
            self.assertEqual(
                executor.submit(_identity, 7).result(
                    timeout=_FUTURE_COMPLETION_TIMEOUT
                ),
                7,
            )
            self.assertTrue(pool._status_receiver_watcher_started.is_set())
        finally:
            pool.shutdown()

        self.assertEqual([proc.exitcode for proc in pool._procs], [0])
        self.assertTrue(in_q._closed)
        self.assertFalse(in_q._joincancelled)
        self.assertTrue(out_q._closed)
        self.assertFalse(out_q._joincancelled)

    def test_shutdown_waits_for_monitor_failure_before_feeder_policy(self) -> None:
        """A late monitor failure still forces non-graceful feeder cleanup."""
        pool = _WorkerPool(
            mp.get_context("spawn"),
            1,
            None,
            (),
            defer_monitor=True,
        )
        monitor = _FailureDuringJoinMonitor(pool._worker_failed)
        pool._monitor = monitor
        in_q = pool._in_q
        out_q = pool._out_q

        pool.shutdown()

        self.assertIsNotNone(monitor.join_timeout)
        self.assertGreater(monitor.join_timeout or 0, 0)
        self.assertTrue(in_q._joincancelled)
        self.assertFalse(out_q._joincancelled)

    def test_monitor_wait_failure_reports_worker_pool_failure(self) -> None:
        """A broken process-handle wait cannot silently disable worker monitoring."""
        pool = object.__new__(_WorkerPool)
        pool._procs = [MagicMock(sentinel=7)]
        pool._shutdown_started = threading.Event()
        pool._worker_failed = threading.Event()
        pool._send_worker_status = MagicMock()

        with (
            patch(
                "spdl.pipeline._subprocess_worker_pool.wait_for_mp_handles",
                side_effect=OSError("wait failed"),
            ),
            self.assertLogs("spdl.pipeline._subprocess_worker_pool", level="ERROR"),
        ):
            pool._monitor_workers()

        self.assertTrue(pool._worker_failed.is_set())
        pool._send_worker_status.assert_called_once_with(
            "Worker pool monitor failed: OSError: wait failed"
        )

    def test_shutdown_does_not_wait_forever_for_stuck_monitor(self) -> None:
        """A delayed monitor neither wedges nor degrades clean worker teardown."""
        ctx = mp.get_context("spawn")
        worker_started = ctx.Event()
        pool = _WorkerPool(
            ctx,
            1,
            _signal_initializer,
            (worker_started,),
            defer_monitor=True,
        )
        self.assertTrue(worker_started.wait(timeout=_PROCESS_READY_TIMEOUT))
        monitor = _StuckMonitor()
        pool._monitor = monitor
        in_q = pool._in_q
        out_q = pool._out_q

        pool.shutdown()

        self.assertIsNotNone(monitor.join_timeout)
        self.assertGreater(monitor.join_timeout or 0, 0)
        self.assertFalse(in_q._joincancelled)
        self.assertFalse(out_q._joincancelled)
        self.assertTrue(pool._worker_status_recv.closed)
        self.assertTrue(pool._worker_status_send.closed)

    def test_status_send_failure_still_closes_pool_resources(self) -> None:
        """Status-channel failures cannot skip queue and endpoint cleanup."""
        pool = _WorkerPool(mp.get_context("spawn"), 1, None, ())
        in_q = pool._in_q
        out_q = pool._out_q

        with patch.object(
            pool,
            "_send_worker_status",
            side_effect=ValueError("status serialization failed"),
        ):
            with self.assertRaisesRegex(ValueError, "status serialization failed"):
                pool.shutdown()

        self.assertTrue(all(not proc.is_alive() for proc in pool._procs))
        self.assertTrue(in_q._closed)
        self.assertTrue(out_q._closed)
        self.assertTrue(pool._worker_status_recv.closed)
        self.assertTrue(pool._worker_status_send.closed)

    def test_worker_exit_after_shutdown_flushes_output_sentinel(self) -> None:
        """A failed worker abandons input but still wakes the result router."""
        ctx = mp.get_context("spawn")
        started = ctx.Event()
        release = ctx.Event()
        pool = _WorkerPool(
            ctx,
            1,
            _exit_during_shutdown_initializer,
            (started, release),
        )
        in_q = pool._in_q
        out_q = pool._out_q
        sentinel_seen = threading.Event()

        def release_after_sentinel() -> None:
            if in_q._reader.poll(_PROCESS_READY_TIMEOUT):
                sentinel_seen.set()
            release.set()

        release_thread = threading.Thread(
            target=release_after_sentinel,
            daemon=True,
        )
        shutdown_started = False
        try:
            self.assertTrue(started.wait(timeout=_PROCESS_READY_TIMEOUT))
            release_thread.start()
            shutdown_started = True
            pool.shutdown()
        finally:
            release.set()
            if release_thread.ident is not None:
                release_thread.join(timeout=_PROCESS_READY_TIMEOUT)
            if not shutdown_started:
                pool.shutdown()

        self.assertTrue(sentinel_seen.is_set())
        self.assertFalse(release_thread.is_alive())
        self.assertEqual([proc.exitcode for proc in pool._procs], [17])
        self.assertTrue(in_q._closed)
        self.assertTrue(in_q._joincancelled)
        self.assertTrue(out_q._closed)
        self.assertFalse(out_q._joincancelled)

    def test_unpicklable_input_fails_its_future_promptly(self) -> None:
        """An invalid submission is reported instead of being dropped by the queue feeder."""
        pool = _WorkerPool(mp.get_context("spawn"), 1, None, ())
        try:
            executor = pool.make_executor()
            future = executor.submit(_identity, _Unpicklable())
            following = executor.submit(_identity, 7)

            with self.assertRaisesRegex(TypeError, "cannot pickle test payload"):
                future.result(timeout=_FUTURE_COMPLETION_TIMEOUT)
            self.assertTrue(future.done())
            self.assertEqual(following.result(timeout=_FUTURE_COMPLETION_TIMEOUT), 7)
        finally:
            _shutdown_pools([pool])
        self.assertTrue(all(not proc.is_alive() for proc in pool._procs))

    def test_input_reducer_base_exception_keeps_feeder_alive(self) -> None:
        """A direct BaseException from a reducer fails one request, not its feeder."""
        pool = _WorkerPool(mp.get_context("spawn"), 1, None, ())
        try:
            executor = pool.make_executor()
            future = executor.submit(_identity, _UnpicklableWithBaseException())
            following = executor.submit(_identity, 9)

            with self.assertRaisesRegex(
                _PicklingBaseException, "direct base exception while pickling"
            ):
                future.result(timeout=_FUTURE_COMPLETION_TIMEOUT)
            self.assertEqual(following.result(timeout=_FUTURE_COMPLETION_TIMEOUT), 9)
        finally:
            _shutdown_pools([pool])

    def test_forced_shutdown_does_not_join_a_backpressured_feeder(self) -> None:
        """Forced teardown discards a feeder blocked behind an unresponsive worker."""
        ctx = mp.get_context("spawn")
        initializer_started = ctx.Event()
        pool = _WorkerPool(ctx, 1, _block_initializer, (initializer_started,))
        shutdown_done = threading.Event()
        self.assertTrue(initializer_started.wait(timeout=_PROCESS_READY_TIMEOUT))

        executor = pool.make_executor()
        serialization_started = threading.Event()
        executor.submit(
            _identity,
            _SerializationSignal(serialization_started, b"x" * (8 * 1024 * 1024)),
        )
        self.assertTrue(serialization_started.wait(timeout=_PROCESS_READY_TIMEOUT))

        def _shutdown() -> None:
            pool.shutdown()
            shutdown_done.set()

        shutdown_thread = threading.Thread(target=_shutdown, daemon=True)
        shutdown_thread.start()
        self.assertTrue(
            shutdown_done.wait(timeout=10),
            "shutdown hung while joining a backpressured queue feeder",
        )
        shutdown_thread.join()
        self.assertTrue(all(not proc.is_alive() for proc in pool._procs))

    def test_output_sentinel_failure_abandons_feeder_and_cleans_up(self) -> None:
        """A failed router wakeup degrades to nonblocking cleanup."""
        ctx = mp.get_context("spawn")
        worker_started = ctx.Event()
        pool = _WorkerPool(ctx, 1, _signal_initializer, (worker_started,))
        self.assertTrue(worker_started.wait(timeout=_PROCESS_READY_TIMEOUT))
        in_q = pool._in_q
        out_q = pool._out_q

        with (
            patch.object(
                out_q,
                "put_nowait",
                side_effect=OSError("output sentinel failed"),
            ),
            self.assertLogs("spdl.pipeline._subprocess_worker_pool", level="WARNING"),
        ):
            pool.shutdown()

        self.assertTrue(all(not proc.is_alive() for proc in pool._procs))
        self.assertTrue(in_q._closed)
        self.assertTrue(out_q._closed)
        self.assertTrue(out_q._joincancelled)

    def test_output_feeder_timeout_keeps_clean_pool_status(self) -> None:
        """A delayed router sentinel does not preempt a dequeued result."""
        pool = _WorkerPool(mp.get_context("spawn"), 1, None, ())
        out_q = pool._out_q
        executor = pool.make_executor()
        gated_queue = _GatedResultQueue(executor._out_q)
        executor._out_q = gated_queue
        future = executor.submit(_identity, 7)
        self.assertTrue(
            gated_queue.result_dequeued.wait(timeout=_FUTURE_COMPLETION_TIMEOUT)
        )
        out_q._wlock.acquire()
        try:
            with patch(
                "spdl.pipeline._subprocess_worker_pool._OUTPUT_FEEDER_JOIN_TIMEOUT",
                0.01,
            ):
                pool.shutdown()

            watcher = executor._worker_watcher
            self.assertIsNotNone(watcher)
            if watcher is not None:
                watcher.join(timeout=5)
                self.assertFalse(watcher.is_alive())
            self.assertTrue(all(proc.exitcode == 0 for proc in pool._procs))
            self.assertTrue(out_q._joincancelled)
            self.assertFalse(future.done())
            self.assertIsNone(executor._broken)

            gated_queue.release_result.set()
            self.assertEqual(
                future.result(timeout=_FUTURE_COMPLETION_TIMEOUT),
                7,
            )
        finally:
            gated_queue.release_result.set()
            out_q._wlock.release()
            feeder = out_q._thread
            if feeder is not None:
                feeder.join(timeout=5)
                self.assertFalse(feeder.is_alive())
            router = executor._thread
            if router is not None:
                router.join(timeout=5)
                self.assertFalse(router.is_alive())

    def test_unpicklable_result_fails_its_future(self) -> None:
        """An invalid worker result becomes an error response instead of a pending future."""
        pool = _WorkerPool(mp.get_context("spawn"), 1, None, ())
        try:
            executor = pool.make_executor()
            future = executor.submit(_return_unpicklable)
            following = executor.submit(_identity, 11)

            with self.assertRaisesRegex(
                RuntimeError, "Worker result could not be serialized"
            ):
                future.result(timeout=_FUTURE_COMPLETION_TIMEOUT)
            self.assertEqual(following.result(timeout=_FUTURE_COMPLETION_TIMEOUT), 11)
        finally:
            _shutdown_pools([pool])

    def test_result_reducer_base_exception_keeps_feeder_alive(self) -> None:
        """A direct BaseException while serializing a result preserves later results."""
        pool = _WorkerPool(mp.get_context("spawn"), 1, None, ())
        try:
            executor = pool.make_executor()
            future = executor.submit(_return_unpicklable_with_base_exception)
            following = executor.submit(_identity, 15)

            with self.assertRaisesRegex(
                RuntimeError,
                "Worker result could not be serialized: "
                "_PicklingBaseException: direct base exception while pickling",
            ):
                future.result(timeout=_FUTURE_COMPLETION_TIMEOUT)
            self.assertEqual(following.result(timeout=_FUTURE_COMPLETION_TIMEOUT), 15)
        finally:
            _shutdown_pools([pool])

    def test_large_payloads_are_serialized_once_off_the_calling_threads(self) -> None:
        """Large requests and results retain one asynchronous serialization pass and order."""
        pool = _WorkerPool(mp.get_context("spawn"), 1, None, ())
        payload = b"x" * (2 * 1024 * 1024)
        input_probe = _SerializationProbe(threading.get_ident(), payload)
        try:
            executor = pool.make_executor()
            futures = [
                executor.submit(_identity, 1),
                executor.submit(_identity, input_probe),
                executor.submit(_return_serialization_probe, payload),
                executor.submit(_identity, 4),
            ]

            self.assertEqual(
                [
                    future.result(timeout=_FUTURE_COMPLETION_TIMEOUT)
                    for future in futures
                ],
                [1, (payload, 1), (payload, 1), 4],
            )
            self.assertEqual(input_probe.calls, 1)
        finally:
            _shutdown_pools([pool])

    def test_unpicklable_exception_fails_its_future(self) -> None:
        """An invalid worker exception is replaced by a serializable error response."""
        pool = _WorkerPool(mp.get_context("spawn"), 1, None, ())
        try:
            executor = pool.make_executor()

            with self.assertRaisesRegex(
                RuntimeError, "Worker exception could not be serialized"
            ):
                executor.submit(_raise_unpicklable).result(
                    timeout=_FUTURE_COMPLETION_TIMEOUT
                )
            self.assertEqual(
                executor.submit(_identity, 13).result(
                    timeout=_FUTURE_COMPLETION_TIMEOUT
                ),
                13,
            )
        finally:
            _shutdown_pools([pool])

    def test_graceful_shutdown_stops_executor_threads(self) -> None:
        """Orderly pool teardown wakes both remote-executor helper threads."""
        pool = _WorkerPool(mp.get_context("spawn"), 1, None, ())
        executor = pool.make_executor()
        try:
            self.assertEqual(
                executor.submit(_identity, 19).result(
                    timeout=_FUTURE_COMPLETION_TIMEOUT
                ),
                19,
            )
            router = executor._thread
            watcher = executor._worker_watcher
            self.assertIsNotNone(router)
            self.assertIsNotNone(watcher)
        finally:
            _shutdown_pools([pool])

        if router is not None:
            router.join(timeout=5)
            self.assertFalse(router.is_alive())
        if watcher is not None:
            watcher.join(timeout=5)
            self.assertFalse(watcher.is_alive())

    def test_graceful_shutdown_preserves_dequeued_result(self) -> None:
        """Orderly watcher shutdown cannot overtake an already-dequeued result."""
        ctx = mp.get_context("spawn")
        worker_started = ctx.Event()
        pool = _WorkerPool(ctx, 1, _signal_initializer, (worker_started,))
        self.assertTrue(worker_started.wait(timeout=_PROCESS_READY_TIMEOUT))
        executor = pool.make_executor()
        gated_queue = _GatedResultQueue(executor._out_q)
        executor._out_q = gated_queue

        try:
            future = executor.submit(_identity, 37)
            self.assertTrue(
                gated_queue.result_dequeued.wait(timeout=_FUTURE_COMPLETION_TIMEOUT)
            )

            pool._send_worker_status(None)
            watcher = executor._worker_watcher
            self.assertIsNotNone(watcher)
            if watcher is not None:
                watcher.join(timeout=5)
                self.assertFalse(watcher.is_alive())
            self.assertFalse(future.done())

            gated_queue.release_result.set()
            self.assertEqual(
                future.result(timeout=_FUTURE_COMPLETION_TIMEOUT),
                37,
            )
        finally:
            gated_queue.release_result.set()
            _shutdown_pools([pool])

        router = executor._thread
        self.assertIsNotNone(router)
        if router is not None:
            router.join(timeout=5)
            self.assertFalse(router.is_alive())

    def test_abrupt_worker_exit_fails_pending_futures_and_shutdown(self) -> None:
        """A dead worker breaks pending work promptly and cannot wedge queue teardown."""
        ctx = mp.get_context("spawn")
        worker_entered = ctx.Event()
        release_worker = ctx.Event()
        submissions_ready = ctx.Event()
        pool = _WorkerPool(
            ctx,
            1,
            _install_exit_gate,
            (worker_entered, release_worker),
            defer_monitor=True,
        )
        receive_status: Any = None
        send_status: Any = None
        client: Any = None
        shutdown_done = threading.Event()

        def _shutdown() -> None:
            _shutdown_pools([pool])
            shutdown_done.set()

        try:
            # Create all secondary IPC under the cleanup guard: a resource-allocation failure
            # here must not leak the worker pool into the rest of the test process.
            receive_status, send_status = ctx.Pipe(duplex=False)
            client = ctx.Process(
                target=_exercise_abrupt_worker_exit,
                args=(
                    pool.make_executor(receiver_is_shared=False),
                    submissions_ready,
                    send_status,
                ),
            )
            # Spawn the submit side before starting the owner monitor, matching
            # ``run_pipeline_in_subprocess`` and proving all liveness state survives spawn.
            client.start()
            send_status.close()
            _start_pool_monitors([pool])
            self.assertTrue(submissions_ready.wait(timeout=_PROCESS_READY_TIMEOUT))
            self.assertTrue(worker_entered.wait(timeout=_PROCESS_READY_TIMEOUT))
            release_worker.set()

            self.assertTrue(
                receive_status.poll(10),
                "spawned submitter remained blocked on Futures after worker exit",
            )
            outcomes = receive_status.recv()
            self.assertEqual(len(outcomes), 3)
            for kind, message in outcomes:
                self.assertEqual(kind, BrokenExecutor.__name__)
                self.assertIn("exited unexpectedly", message)
            client.join(timeout=5)
            self.assertFalse(client.is_alive())
        finally:
            release_worker.set()
            if client is not None and client.pid is not None and client.is_alive():
                client.terminate()
                client.join(timeout=5)
            if receive_status is not None:
                receive_status.close()
            if send_status is not None:
                send_status.close()

            shutdown_thread = threading.Thread(target=_shutdown, daemon=True)
            shutdown_thread.start()
            self.assertTrue(
                shutdown_done.wait(timeout=5),
                "shutdown hung after an abrupt worker exit",
            )
            shutdown_thread.join()
            self.assertTrue(pool._worker_status_recv.closed)

    @unittest.skipUnless("fork" in mp.get_all_start_methods(), "requires fork")
    def test_submitter_exit_does_not_deadlock_owner_shutdown(self) -> None:
        """A terminated remote watcher cannot wedge the owner's stop notification."""
        ctx = mp.get_context("fork")
        pool = _WorkerPool(ctx, 1, None, (), defer_monitor=True)
        receive_status: Any = None
        send_status: Any = None
        client: Any = None
        shutdown_done = threading.Event()

        def _shutdown() -> None:
            _shutdown_pools([pool])
            shutdown_done.set()

        try:
            receive_status, send_status = ctx.Pipe(duplex=False)
            client = ctx.Process(
                target=_submit_then_exit,
                args=(pool.make_executor(receiver_is_shared=False), send_status),
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"This process \(pid=\d+\) is multi-threaded,.*",
                    category=DeprecationWarning,
                )
                client.start()
            send_status.close()
            _start_pool_monitors([pool])

            self.assertTrue(
                receive_status.poll(5),
                "forked submitter did not complete its worker-pool request",
            )
            self.assertEqual(receive_status.recv(), 31)
            client.join(timeout=5)
            self.assertFalse(client.is_alive())
        finally:
            if client is not None and client.pid is not None and client.is_alive():
                client.terminate()
                client.join(timeout=5)
            if receive_status is not None:
                receive_status.close()
            if send_status is not None:
                send_status.close()

            shutdown_thread = threading.Thread(target=_shutdown, daemon=True)
            shutdown_thread.start()
            self.assertTrue(
                shutdown_done.wait(timeout=5),
                "shutdown deadlocked after the remote watcher process exited",
            )
            shutdown_thread.join()
            self.assertTrue(pool._worker_status_recv.closed)
