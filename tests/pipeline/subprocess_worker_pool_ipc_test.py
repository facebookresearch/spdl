# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for failures in the hoisted worker-pool IPC."""

import errno
import multiprocessing as mp
import queue
import threading
import unittest
from concurrent.futures import BrokenExecutor, Future
from typing import Any
from unittest.mock import MagicMock, patch

from spdl.pipeline._subprocess_worker_pool import (
    _handle_queue_feeder_error,
    _RemoteExecutor,
    _Result,
    _shutdown_pools,
    _WorkerPool,
)

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
                executor = _RemoteExecutor(in_q, MagicMock(), 1)
                futures = [Future(), Future()]
                executor._futures.update({3: futures[0], 4: futures[1]})

                def fail_callback(_: Future[Any]) -> None:
                    raise _PicklingBaseException("callback failed")

                futures[0].add_done_callback(fail_callback)
                with (
                    patch(
                        "spdl.pipeline._subprocess_worker_pool.traceback.print_exception"
                    ) as print_exception,
                    patch(
                        "spdl.pipeline._subprocess_worker_pool.traceback.print_exc"
                    ) as print_exc,
                ):
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
                print_exc.assert_called_once_with()
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

    def test_submission_failure_contains_callback_base_exception(self) -> None:
        """A failing done callback cannot escape the queue-feeder error path."""
        executor = _RemoteExecutor(MagicMock(), MagicMock(), 1)
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
