# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import errno
import gc
import multiprocessing as mp
import os
import queue
import sys
import threading
import unittest
import weakref
from typing import Any, cast
from unittest.mock import MagicMock, patch

from spdl.pipeline import _pool_cleanup, build_pipeline, PipelineBuilder
from spdl.pipeline._components import _POOL_SHUTDOWN
from spdl.pipeline._pipeline import _PipelineImpl
from spdl.pipeline._subprocess_pipeline_pool import (
    _InterpreterBackend,
    _InterpreterWorker,
    _PoolBackend,
    _ProcessBackend,
    _ProcessWorker,
    _shutdown_pipeline_pools,
    _SubprocessPipelinePool,
    _Worker,
)
from spdl.pipeline.config import (
    get_pending_pipeline_pool_cleanup_count,
    retry_pending_pipeline_pool_cleanup,
)
from spdl.pipeline.defs import PipelineConfig, SinkConfig, SourceConfig

from . import build_failure_cleanup_test as _build_failure_tests


class _WorkerStub(_Worker):
    def __init__(self) -> None:
        self.fail: BaseException | None = None
        self.calls = 0
        self.callback: Any = None

    def stop(self) -> None:
        self.calls += 1
        if self.callback is not None:
            self.callback()
        if self.fail is not None:
            raise self.fail


class _BackendStub(_PoolBackend):
    def __init__(self, workers: list[_WorkerStub]) -> None:
        self.workers = workers
        self.spawned = 0
        self.queue_count = 0
        self.closed: list[Any] = []
        self.markers: list[Any] = []
        self.close_fail: Any = None
        self.spawn_fail_at = -1
        self.queue_fail_at = -1
        self.close_callback: Any = None

    def make_serialization_failure_flag(self) -> None:
        return None

    def make_queue(self, maxsize: int) -> Any:
        if self.queue_count == self.queue_fail_at:
            raise ValueError("queue construction failed")
        self.queue_count += 1
        return queue.Queue(maxsize)

    def spawn(
        self,
        target: Any,
        args: Any,
        register: Any,
    ) -> _Worker:
        if self.spawned == self.spawn_fail_at:
            raise ValueError("worker construction failed")
        worker = self.workers[self.spawned]
        register(worker)
        self.spawned += 1
        return worker

    def try_put_shutdown(self, q: Any) -> None:
        self.markers.append(q)

    def close_queue(self, q: Any) -> None:
        if self.close_callback is not None:
            self.close_callback()
        if q is self.close_fail:
            raise OSError("queue close failed")
        self.closed.append(q)


def _pool(backend: _PoolBackend, n: int) -> _SubprocessPipelinePool:
    return _SubprocessPipelinePool(
        backend, n, PipelineConfig(SourceConfig([]), [], SinkConfig(1)), {}, None, ()
    )


def _wait_for_release(ready: Any, release: Any) -> None:
    ready.set()
    if not release.wait(30):
        raise TimeoutError("test worker did not receive release")


def _fill_input_pipe(q: Any, ready: Any) -> None:
    send_bytes = q._send_bytes

    def send(data: bytes) -> None:
        ready.set()
        send_bytes(data)

    q._send_bytes = send
    q.put(b"x" * (4 * 1024 * 1024))
    q.close()
    q.join_thread()


def _wait_for_interpreter_release(ready: Any, release: Any) -> None:
    ready.put(True)
    release.get(timeout=10)


class _RealProcessBackend(_ProcessBackend):
    def __init__(self, ctx: Any, ready: Any, release: Any) -> None:
        super().__init__(ctx)
        self.ready = ready
        self.release = release

    def spawn(
        self,
        target: Any,
        args: Any,
        register: Any,
    ) -> _Worker:
        return super().spawn(_wait_for_release, (self.ready, self.release), register)


class PoolCleanupTest(unittest.TestCase):
    def setUp(self) -> None:
        with _pool_cleanup._LOCK:
            outer = _pool_cleanup._PENDING
            self.addCleanup(self._finish_pending, outer)
            _pool_cleanup._PENDING = {}

    def _finish_pending(self, outer: dict[tuple[int, int], Any]) -> None:
        try:
            for pool in _pool_cleanup._snapshot():
                if isinstance(pool, _SubprocessPipelinePool):
                    for worker in pool._workers:
                        if isinstance(worker, _WorkerStub):
                            worker.fail = None
                            worker.callback = None
                    if isinstance(pool._backend, _BackendStub):
                        pool._backend.close_fail = None
                        pool._backend.close_callback = None
            retry_pending_pipeline_pool_cleanup()
            self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 0)
        finally:
            # A failed test cleanup must not lose the unfinished resource roots
            # when the fixture restores the outer shared-runner registry.
            with _pool_cleanup._LOCK:
                scoped = _pool_cleanup._PENDING
                _pool_cleanup._PENDING = outer
                outer.update(scoped)

    def test_worker_failure_retains_all_queues_and_retries_only_failed_worker(
        self,
    ) -> None:
        """No queue closes before every worker retires, and completed stops are not repeated."""
        first, second = _WorkerStub(), _WorkerStub()
        first.fail = TimeoutError("still alive")
        backend = _BackendStub([first, second])
        pool = _pool(backend, 2)
        with self.assertRaises(ExceptionGroup):
            pool.shutdown()
        self.assertEqual(pool._workers, [first])
        self.assertEqual(backend.closed, [])
        self.assertFalse(pool._retired)
        self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 1)
        with self.assertRaisesRegex(RuntimeError, "shutdown has started"):
            pool.make_handle()
        first.fail = None
        retry_pending_pipeline_pool_cleanup()
        self.assertEqual((first.calls, second.calls), (2, 1))
        self.assertEqual(len(backend.closed), 3)
        self.assertEqual(pool._workers, [])
        self.assertEqual(pool._in_qs, [])
        self.assertIsNone(pool._out_q)
        self.assertTrue(pool._retired)
        pool.shutdown()
        self.assertEqual((first.calls, second.calls), (2, 1))
        self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 0)

    def test_failed_close_keeps_only_its_queue_for_retry(self) -> None:
        """Queue-close failure cannot repeat successful worker retirement or other closes."""
        worker = _WorkerStub()
        backend = _BackendStub([worker])
        pool = _pool(backend, 1)
        failed = pool._in_qs[0]
        backend.close_fail = failed
        with self.assertRaises(ExceptionGroup):
            pool.shutdown()
        self.assertEqual(worker.calls, 1)
        self.assertEqual(pool._in_qs, [failed])
        self.assertIsNone(pool._out_q)
        self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 1)
        backend.close_fail = None
        pool.shutdown()
        self.assertEqual(worker.calls, 1)
        self.assertEqual(len(backend.closed), 2)
        self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 0)

    def test_drop_owner_preserves_retry_root_then_releases_it(self) -> None:
        """Registry roots keep unfinished workers alive until a successful retry."""
        worker = _WorkerStub()
        worker.fail = RuntimeError("stop failed")
        pool = _pool(_BackendStub([worker]), 1)
        ref = weakref.ref(pool)
        with self.assertRaises(ExceptionGroup):
            pool.shutdown()
        del pool
        gc.collect()
        self.assertIsNotNone(ref())
        worker.fail = None
        retry_pending_pipeline_pool_cleanup()
        gc.collect()
        self.assertIsNone(ref())
        self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 0)

    def test_reentrant_retry_does_not_deadlock_or_repeat_active_stop(self) -> None:
        """Registry inspection is safe in callbacks; reentrant shutdown reports in-progress."""
        worker = _WorkerStub()
        pool = _pool(_BackendStub([worker]), 1)
        observed = []

        def callback() -> None:
            observed.append(get_pending_pipeline_pool_cleanup_count())
            with self.assertRaises(ExceptionGroup):
                retry_pending_pipeline_pool_cleanup()

        worker.callback = callback
        pool.shutdown()
        self.assertEqual(observed, [1])
        self.assertEqual(worker.calls, 1)
        self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 0)

    def test_concurrent_shutdown_does_not_publish_early_success(self) -> None:
        """An overlapping call cannot report success while worker cleanup is active."""
        worker = _WorkerStub()
        entered, release = threading.Event(), threading.Event()
        worker.callback = lambda: (entered.set(), release.wait(3))
        pool = _pool(_BackendStub([worker]), 1)
        errors = []

        def shutdown() -> None:
            try:
                pool.shutdown()
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=shutdown)
        thread.start()
        try:
            self.assertTrue(entered.wait(3))
            with self.assertRaisesRegex(RuntimeError, "already in progress"):
                pool.shutdown()
            self.assertFalse(pool._retired)
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(worker.calls, 1)

    def test_constructor_rollback_retains_started_worker_and_primary_error(
        self,
    ) -> None:
        """Partial spawn failure preserves its primary error and roots unfinished cleanup."""
        worker = _WorkerStub()
        worker.fail = TimeoutError("cannot join")
        backend = _BackendStub([worker])
        backend.spawn_fail_at = 1
        with self.assertLogs("spdl.pipeline", level="ERROR"):
            with self.assertRaisesRegex(ValueError, "worker construction failed"):
                _pool(backend, 2)
        self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 1)
        self.assertEqual(backend.closed, [])
        worker.fail = None
        retry_pending_pipeline_pool_cleanup()
        self.assertEqual(len(backend.closed), 3)
        self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 0)

    def test_interpreter_close_warning_retains_pool_until_success(self) -> None:
        """Logging a failed close never retires the worker or its dependent queues."""
        interp, thread = MagicMock(), MagicMock()
        thread.is_alive.return_value = False
        interp.close.side_effect = RuntimeError("close failed")
        worker = _InterpreterWorker(interp, thread)
        driver = _WorkerStub()
        driver.callback = worker.stop
        backend = _BackendStub([driver])
        pool = _pool(backend, 1)
        with self.assertLogs(
            "spdl.pipeline._subprocess_pipeline_pool", level="WARNING"
        ) as logs:
            with self.assertRaises(ExceptionGroup):
                pool.shutdown()
        self.assertIn("cleanup remains pending", logs.output[0])
        self.assertFalse(worker._closed)
        self.assertEqual(pool._workers, [driver])
        self.assertEqual(backend.closed, [])
        self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 1)
        interp.close.side_effect = None
        retry_pending_pipeline_pool_cleanup()
        self.assertTrue(worker._closed)
        self.assertTrue(pool._retired)
        self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 0)
        self.assertEqual(len(backend.closed), 2)
        worker.stop()
        pool.shutdown()
        self.assertEqual(interp.close.call_count, 2)

    def test_partial_queue_creation_releases_acquired_queues(self) -> None:
        """Queue creation rollback includes queues created before any workers exist."""
        backend = _BackendStub([])
        backend.queue_fail_at = 2
        with self.assertRaisesRegex(ValueError, "queue construction failed"):
            _pool(backend, 2)
        self.assertEqual(len(backend.closed), 2)
        self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 0)

    def test_base_exception_does_not_destroy_remaining_roots(self) -> None:
        """An interruption leaves the entire unfinished cleanup path discoverable."""
        worker = _WorkerStub()
        worker.fail = KeyboardInterrupt()
        pool = _pool(_BackendStub([worker]), 1)
        with self.assertRaises(KeyboardInterrupt):
            pool.shutdown()
        self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 1)
        worker.fail = None
        retry_pending_pipeline_pool_cleanup()
        self.assertTrue(pool._retired)

    def test_pool_list_cleanup_attempts_later_pools(self) -> None:
        """One failed pool cannot prevent independent pools from being reaped."""
        worker = _WorkerStub()
        worker.fail = RuntimeError("failed")
        first = _pool(_BackendStub([worker]), 1)
        second = _pool(_BackendStub([_WorkerStub()]), 1)
        with self.assertRaises(ExceptionGroup):
            _shutdown_pipeline_pools([first, second])
        self.assertTrue(second._retired)
        self.assertFalse(first._retired)
        self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 1)

    def test_pipeline_keeps_failed_pool_until_next_cleanup(self) -> None:
        """Pipeline cleanup drops successful pools but retains failed ones for retry."""
        first, second = MagicMock(), MagicMock()
        first.shutdown.side_effect = RuntimeError("failed")
        impl = object.__new__(_PipelineImpl)
        impl._pools = [first, second]
        impl._pool_shutdown_lock = threading.Lock()
        with self.assertLogs("spdl.pipeline", level="WARNING"):
            impl._shutdown_pools()
        self.assertEqual(impl._pools, [first])
        first.shutdown.side_effect = None
        impl._shutdown_pools()
        self.assertEqual(impl._pools, [])
        self.assertEqual(first.shutdown.call_count, 2)
        self.assertEqual(second.shutdown.call_count, 1)

    def test_later_build_failure_retains_fused_cleanup(self) -> None:
        """A resource already acquired by build remains retryable after a later build error."""
        worker = _WorkerStub()
        worker.fail = RuntimeError("cannot reap")
        pool = _pool(_BackendStub([worker]), 1)
        cfg = PipelineBuilder().add_source([1]).add_sink(1).get_config()
        with (
            patch("spdl.pipeline._build._DEFAULT_BUILD_CALLBACK", None),
            patch(
                "spdl.pipeline._build._fuse_marked_regions", return_value=(cfg, [pool])
            ),
            self.assertLogs("spdl.pipeline", level="ERROR"),
        ):
            with self.assertRaises(ValueError):
                build_pipeline(cfg, num_threads=0)
        self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 1)
        worker.fail = None
        retry_pending_pipeline_pool_cleanup()
        self.assertTrue(pool._retired)

    def test_registry_allocation_can_reenter_cleanup_during_gc(self) -> None:
        """A GC finalizer reached during registry bookkeeping cannot deadlock."""
        worker = _WorkerStub()
        pool = _pool(_BackendStub([worker]), 1)
        finalized = []

        class Cycle:
            def __init__(self) -> None:
                self.cycle = self

        target = Cycle()
        finalizer = weakref.finalize(target, pool.shutdown)
        was_enabled = gc.isenabled()
        gc.disable()
        del target

        def collect_once(frame: Any, event: str, arg: Any) -> Any:
            if (
                event == "line"
                and frame.f_code.co_name == "_snapshot"
                and frame.f_globals.get("__name__") == _pool_cleanup.__name__
                and cast(Any, _pool_cleanup._LOCK)._is_owned()
                and not finalized
            ):
                finalized.append(True)
                gc.collect()
            return collect_once

        try:
            sys.settrace(collect_once)
            _pool_cleanup._snapshot()
        finally:
            sys.settrace(None)
            if was_enabled:
                gc.enable()
            finalizer()
        self.assertEqual(finalized, [True])
        self.assertTrue(pool._retired)
        self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 0)

    def test_batch_interruption_attempts_and_retains_all_owners(self) -> None:
        """A cleanup interruption cannot lose unattempted independent pools."""
        first_worker, second_worker = _WorkerStub(), _WorkerStub()
        first_worker.fail = KeyboardInterrupt()
        first = _pool(_BackendStub([first_worker]), 1)
        second = _pool(_BackendStub([second_worker]), 1)
        with self.assertRaises(BaseExceptionGroup):
            _shutdown_pipeline_pools([first, second])
        self.assertEqual(second_worker.calls, 1)
        self.assertTrue(second._retired)
        self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 1)
        first_worker.fail = None
        retry_pending_pipeline_pool_cleanup()
        self.assertTrue(first._retired)

    def test_pipeline_cleanup_reentrancy_preserves_primary_bookkeeping(self) -> None:
        """A reentrant owner pass does not retire/remove a pool twice."""
        first, second = MagicMock(), MagicMock()
        impl = object.__new__(_PipelineImpl)
        impl._pools = [first, second]
        impl._pool_shutdown_lock = threading.Lock()
        first.shutdown.side_effect = impl._shutdown_pools
        with self.assertLogs("spdl.pipeline", level="WARNING"):
            impl._shutdown_pools()
        self.assertEqual(impl._pools, [])
        self.assertEqual(first.shutdown.call_count, 1)
        self.assertEqual(second.shutdown.call_count, 1)
        self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 0)

    def test_interpreter_owner_is_registered_before_physical_start(self) -> None:
        """Interrupted interpreter startup leaves its worker and queues retryable."""
        interp = MagicMock()
        interp.call_in_thread.side_effect = ValueError(
            "startup failed after thread start"
        )
        interp.close.side_effect = RuntimeError("thread still alive")
        backend = object.__new__(_InterpreterBackend)
        backend._interpreters = MagicMock()
        backend._interpreters.create.return_value = interp
        with self.assertLogs("spdl.pipeline", level="ERROR"):
            with self.assertRaisesRegex(ValueError, "after thread start"):
                _pool(backend, 1)
        self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 1)
        pending = _pool_cleanup._snapshot()
        self.assertEqual(len(pending), 1)
        owner = pending[0]
        assert isinstance(owner, _SubprocessPipelinePool)
        self.assertEqual(len(owner._workers), 1)
        self.assertIsInstance(owner._workers[0], _InterpreterWorker)
        self.assertTrue(owner._in_qs[0].put.called)
        interp.close.side_effect = None
        retry_pending_pipeline_pool_cleanup()
        self.assertTrue(owner._retired)
        self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 0)

    def test_process_owner_is_registered_before_physical_start(self) -> None:
        """A post-start interruption cannot close queues or lose the process owner."""
        proc = MagicMock(pid=None, exitcode=None)

        def start() -> None:
            proc.pid = 1234
            raise MemoryError("interrupted after process start")

        proc.start.side_effect = start
        ctx = MagicMock()
        ctx.Process.return_value = proc

        def make_queue(*args: Any, **kwargs: Any) -> Any:
            q = MagicMock()
            q._thread = None
            return q

        ctx.Queue.side_effect = make_queue
        backend = _ProcessBackend(ctx)
        with self.assertLogs("spdl.pipeline", level="ERROR"):
            with self.assertRaisesRegex(MemoryError, "after process start"):
                _pool(backend, 1)
        self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 1)
        owner = _pool_cleanup._snapshot()[0]
        assert isinstance(owner, _SubprocessPipelinePool)
        self.assertEqual(len(owner._workers), 1)
        registered = owner._workers[0]
        assert isinstance(registered, _ProcessWorker)
        self.assertIs(registered._proc, proc)
        for q in (*owner._in_qs, owner._out_q):
            q.close.assert_not_called()
        proc.exitcode = 0
        retry_pending_pipeline_pool_cleanup()
        self.assertTrue(owner._retired)
        self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 0)

    def test_foreign_batch_cleanup_does_not_publish_child_obligation(self) -> None:
        """An inherited pool is rejected before rooting, without skipping local pools."""
        first, second = _WorkerStub(), _WorkerStub()
        foreign = _pool(_BackendStub([first]), 1)
        local = _pool(_BackendStub([second]), 1)
        foreign._owner_pid = os.getpid() + 1
        try:
            with self.assertRaises(BaseExceptionGroup):
                _shutdown_pipeline_pools([foreign, local])
            self.assertEqual(first.calls, 0)
            self.assertEqual(second.calls, 1)
            self.assertTrue(local._retired)
            self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 0)
        finally:
            foreign._owner_pid = os.getpid()
            foreign.shutdown()

    def test_foreign_pipeline_cleanup_does_not_publish_child_obligation(self) -> None:
        """An inherited Pipeline cannot turn parent-owned pools into local retry roots."""
        worker = _WorkerStub()
        pool = _pool(_BackendStub([worker]), 1)
        impl = object.__new__(_PipelineImpl)
        impl._pools = [pool]
        impl._pool_shutdown_lock = threading.Lock()
        pool._owner_pid = os.getpid() + 1
        try:
            with self.assertLogs("spdl.pipeline", level="WARNING"):
                impl._shutdown_pools()
            self.assertEqual(worker.calls, 0)
            self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 0)
        finally:
            pool._owner_pid = os.getpid()
            impl._shutdown_pools()

    @unittest.skipIf(
        sys.platform == "win32", "Windows queues have no shared write lock"
    )
    def test_shutdown_does_not_start_feeder_on_abandoned_writer_lock(self) -> None:
        """A killed remote producer cannot strand a new owner shutdown feeder."""
        ctx = mp.get_context("spawn")
        worker_ready, worker_release = ctx.Event(), ctx.Event()
        backend = _RealProcessBackend(ctx, worker_ready, worker_release)
        pool = _pool(backend, 1)
        q: Any = pool._in_qs[0]
        worker = cast(_ProcessWorker, pool._workers[0])._proc
        ready = ctx.Event()
        producer = ctx.Process(target=_fill_input_pipe, args=(q, ready))
        stranded = False
        try:
            self.assertTrue(worker_ready.wait(10))
            producer.start()
            self.assertTrue(ready.wait(10))
            producer.terminate()
            producer.join(5)
            self.assertIsNotNone(producer.exitcode)
            acquired = q._wlock.acquire(False)
            if acquired:
                q._wlock.release()
            stranded = not acquired
            self.assertTrue(stranded)
            self.assertIsNone(q._thread)
            backend.try_put_shutdown(q)
            self.assertIsNone(q._thread)
            self.assertFalse(q._wlock.acquire(False))
            pool.shutdown()
            self.assertIsNotNone(worker.exitcode)
            self.assertTrue(pool._retired)
            self.assertEqual(pool._workers, [])
            self.assertEqual(pool._in_qs, [])
            self.assertIsNone(pool._out_q)
            self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 0)
            self.assertTrue(q._reader.closed)
            self.assertTrue(q._writer.closed)
        finally:
            if worker.exitcode is None:
                worker_release.set()
            self._rescue_input_queue(producer, q, stranded)

    def _rescue_input_queue(self, producer: Any, q: Any, stranded: bool) -> None:
        """Retire the deliberately stranded fixture even against the unfixed parent."""
        if producer.pid is not None:
            if producer.is_alive():
                producer.kill()
                producer.join(5)
            producer.close()
        q.cancel_join_thread()
        q._reader.close()
        if stranded:
            q._wlock.release()
        q.close()
        if q._thread is not None:
            q._thread.join(3)
            self.assertFalse(q._thread.is_alive())
        q._writer.close()

    def test_shutdown_marker_still_reaches_healthy_queue(self) -> None:
        """Best-effort shutdown still delivers with a new or an existing feeder."""
        ctx = mp.get_context("spawn")
        for warm in (False, True):
            with self.subTest(warm=warm):
                q: Any = ctx.Queue(2)
                backend = _ProcessBackend(ctx)
                try:
                    if warm:
                        q.put("warm")
                        self.assertEqual(q.get(timeout=3), "warm")
                    backend.try_put_shutdown(q)
                    self.assertEqual(q.get(timeout=3), (_POOL_SHUTDOWN, None))
                finally:
                    backend.close_queue(q)
                self.assertFalse(q._thread.is_alive())

    def test_shutdown_write_lock_errors_remain_visible(self) -> None:
        """Only ordinary lock contention can skip a shutdown marker."""
        for operation in ("acquire", "release"):
            with self.subTest(operation=operation):
                q = MagicMock()
                q._thread = None
                q._wlock.acquire.return_value = True
                getattr(q._wlock, operation).side_effect = OSError("lock failed")
                with self.assertRaisesRegex(OSError, "lock failed"):
                    _ProcessBackend(None).try_put_shutdown(q)
                q.put_nowait.assert_not_called()
                if operation == "acquire":
                    q._wlock.release.assert_not_called()

    def test_shutdown_marker_without_write_lock(self) -> None:
        """A queue without a write lock keeps its ordinary best-effort put path."""
        q = MagicMock()
        q._thread = None
        q._wlock = None
        backend = _ProcessBackend(None)
        backend.try_put_shutdown(q)
        q.put_nowait.assert_called_once_with((_POOL_SHUTDOWN, None))
        q.put_nowait.side_effect = queue.Full
        backend.try_put_shutdown(q)
        self.assertEqual(q.put_nowait.call_count, 2)

    def test_queue_reader_close_precedes_feeder_sentinel(self) -> None:
        """Concurrent endpoint close cannot consume the sentinel and strand the feeder."""
        ctx = mp.get_context("spawn")
        q: Any = ctx.Queue(2)
        caller = threading.current_thread()
        closing, feeder_seen = threading.Event(), threading.Event()
        close_reader = q._reader.close
        errors: list[tuple[type[Exception], int | None]] = []

        def guarded_close() -> None:
            if threading.current_thread() is caller:
                if q._closed:
                    closing.set()
                    if not feeder_seen.wait(2):
                        raise TimeoutError("feeder did not reach its reader close")
            elif closing.is_set():
                feeder_seen.set()
                raise OSError(errno.EBADF, "reader close raced the feeder")
            close_reader()

        q._reader.close = guarded_close
        q._on_queue_feeder_error = lambda error, obj: errors.append(
            (type(error), getattr(error, "errno", None))
        )
        q.put(1)
        try:
            _ProcessBackend(ctx).close_queue(q)
            self.assertFalse(q._thread.is_alive())
            self.assertFalse(closing.is_set())
            self.assertFalse(
                any(
                    issubclass(error_type, OSError) and code == errno.EBADF
                    for error_type, code in errors
                )
            )
        finally:
            closing.clear()
            close_reader()
            q.close()
            # Rescue the deliberately broken old ordering when checking the
            # regression against its parent, without leaving a daemon feeder.
            if q._thread.is_alive():
                type(q)._finalize_close(q._buffer, q._notempty)
            q._thread.join(2)
            q._writer.close()

    def test_process_queue_feeder_must_finish_before_owner_release(self) -> None:
        """Queue close cannot report success while a bounded feeder send is still active."""
        ctx = mp.get_context("spawn")
        q: Any = ctx.Queue(2)
        entered, release = threading.Event(), threading.Event()
        original = q._send_bytes

        def send(data: bytes) -> None:
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test feeder release timed out")
            original(data)

        q._send_bytes = send
        feeder_errors: list[type[Exception]] = []
        q._on_queue_feeder_error = lambda error, obj: feeder_errors.append(type(error))
        backend = _ProcessBackend(ctx)
        q.put(1)
        try:
            self.assertTrue(entered.wait(3))
            with patch(
                "spdl.pipeline._subprocess_pipeline_pool._QUEUE_FEEDER_JOIN_TIMEOUT",
                0.001,
            ):
                with self.assertRaises(TimeoutError):
                    backend.close_queue(q)
            self.assertTrue(q._thread.is_alive())
            self.assertTrue(q._reader.closed)
            self.assertFalse(q._writer.closed)
            release.set()
            backend.close_queue(q)
            self.assertFalse(q._thread.is_alive())
            self.assertTrue(feeder_errors)
            self.assertTrue(issubclass(feeder_errors[0], BrokenPipeError))
            self.assertTrue(q._reader.closed)
            self.assertTrue(q._writer.closed)
        finally:
            release.set()
            q.close()
            q._thread.join(3)
            q._reader.close()
            q._writer.close()

    if sys.version_info >= (3, 14):

        def test_real_interpreter_failed_close_then_one_success(self) -> None:
            """A native busy-close rejection preserves the interpreter for a later retry."""
            api = _InterpreterBackend()._interpreters
            interp = api.create()
            ready, release = api.create_queue(), api.create_queue()
            thread = interp.call_in_thread(
                _wait_for_interpreter_release, ready, release
            )
            worker = _InterpreterWorker(interp, None)
            try:
                self.assertTrue(ready.get(timeout=5))
                with patch.object(interp, "close", wraps=interp.close) as close:
                    with self.assertLogs(
                        "spdl.pipeline._subprocess_pipeline_pool", level="WARNING"
                    ):
                        with self.assertRaises(api.InterpreterError):
                            worker.stop()
                    self.assertFalse(worker._closed)
                    self.assertTrue(interp.is_running())
                    self.assertIn(interp.id, [item.id for item in api.list_all()])
                    release.put(True)
                    thread.join(5)
                    self.assertFalse(thread.is_alive())
                    worker._thread = thread
                    worker.stop()
                    worker.stop()
                    self.assertTrue(worker._closed)
                    self.assertNotIn(interp.id, [item.id for item in api.list_all()])
                    self.assertEqual(close.call_count, 2)
            finally:
                release.put(True)
                thread.join(5)
                worker._thread = thread
                worker.stop()

        def test_real_interpreter_worker_retirement(self) -> None:
            """Actual interpreter pools close their worker and release queues repeatedly."""
            for _ in range(2):
                pool = _SubprocessPipelinePool(
                    _InterpreterBackend(),
                    1,
                    PipelineConfig(SourceConfig([]), [], SinkConfig(1)),
                    {"num_threads": 2},
                    None,
                    (),
                )
                worker = pool._workers[0]
                assert isinstance(worker, _InterpreterWorker)
                try:
                    pool.shutdown()
                    self.assertFalse(worker._thread.is_alive())
                    self.assertTrue(worker._closed)
                    self.assertTrue(pool._retired)
                    self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 0)
                finally:
                    pool.shutdown()

    def test_real_process_reap_failure_is_retryable_after_owner_is_dropped(
        self,
    ) -> None:
        """A live spawned process and queues remain owned after injected join failure."""
        ctx = mp.get_context("spawn")
        ready, release = ctx.Event(), ctx.Event()
        backend = _RealProcessBackend(ctx, ready, release)
        pool = _pool(backend, 1)
        worker = pool._workers[0]
        assert isinstance(worker, _ProcessWorker)
        proc = worker._proc
        self.addCleanup(lambda: proc.kill() if proc.is_alive() else None)
        self.addCleanup(proc.join, 5)
        ref = weakref.ref(pool)
        try:
            self.assertTrue(ready.wait(10))
            with patch.object(
                proc, "join", side_effect=RuntimeError("injected join failure")
            ):
                with self.assertRaises(ExceptionGroup):
                    pool.shutdown()
            self.assertTrue(proc.is_alive())
            self.assertFalse(pool._out_q._closed)
            del pool
            gc.collect()
            self.assertIsNotNone(ref())
            release.set()
            retry_pending_pipeline_pool_cleanup()
            self.assertFalse(proc.is_alive())
            gc.collect()
            self.assertIsNone(ref())
            self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 0)
        finally:
            release.set()
            retry_pending_pipeline_pool_cleanup()

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork")
    def test_fork_child_never_retries_parent_roots_or_inherited_lock(self) -> None:
        """A fork while registry lock is held cannot deadlock or stop parent workers."""
        worker = _WorkerStub()
        worker.fail = RuntimeError("failed")
        pool = _pool(_BackendStub([worker]), 1)
        with self.assertRaises(ExceptionGroup):
            pool.shutdown()
        ctx = mp.get_context("fork")
        result: Any = ctx.Queue()

        def child() -> None:
            result.put(get_pending_pipeline_pool_cleanup_count())
            retry_pending_pipeline_pool_cleanup()

        with _pool_cleanup._LOCK:
            proc = ctx.Process(target=child)
            proc.start()
        try:
            self.assertEqual(result.get(timeout=5), 0)
            proc.join(5)
            self.assertEqual(proc.exitcode, 0)
            self.assertEqual(worker.calls, 1)
        finally:
            if proc.is_alive():
                proc.kill()
            proc.join(5)
            result.close()
            result.join_thread()


class CleanupRegistryIsolationTest(unittest.TestCase):
    def test_build_failure_fixture_retires_its_mock_roots(self) -> None:
        """An earlier build-failure test cannot contaminate the next shared-runner test."""
        before = {id(owner) for owner in _pool_cleanup._snapshot()}
        case = _build_failure_tests.BuildFailureCleanupTest(
            "test_constructor_failure_closes_all_unowned_resources"
        )
        result = unittest.TestResult()
        case.run(result)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.failures, [])
        self.assertEqual({id(owner) for owner in _pool_cleanup._snapshot()}, before)

    def test_fixture_preserves_unrelated_pending_owner(self) -> None:
        """Local cleanup assertions neither retry nor discard an outer pending owner."""
        original = _pool_cleanup._PENDING
        before = {id(owner) for owner in _pool_cleanup._snapshot()}
        worker = _WorkerStub()
        worker.fail = TimeoutError("unrelated owner still alive")
        owner = _pool(_BackendStub([worker]), 1)
        try:
            with self.assertRaises(ExceptionGroup):
                owner.shutdown()
            result = unittest.TestResult()
            PoolCleanupTest("test_base_exception_does_not_destroy_remaining_roots").run(
                result
            )
            self.assertEqual(result.errors, [])
            self.assertEqual(result.failures, [])
            self.assertIs(_pool_cleanup._PENDING, original)
            self.assertEqual(worker.calls, 1)
            self.assertFalse(owner._retired)
            self.assertEqual(
                {id(pool) for pool in _pool_cleanup._snapshot()}, before | {id(owner)}
            )
        finally:
            worker.fail = None
            owner.shutdown()

    def test_fixture_finalizer_registration_uses_restored_registry(self) -> None:
        """Finalizers after cleanup publication cannot register into a discarded fixture."""
        original = _pool_cleanup._PENDING

        class FollowupOwner:
            def shutdown(self) -> None:
                _pool_cleanup._release_pool_cleanup(self)

        followup = FollowupOwner()

        class RetiringOwner:
            def shutdown(self) -> None:
                _pool_cleanup._release_pool_cleanup(self)

        class FinalizerCleanup(PoolCleanupTest):
            def runTest(self) -> None:
                retiring = RetiringOwner()
                weakref.finalize(retiring, _pool_cleanup._retain_pool_cleanup, followup)
                _pool_cleanup._retain_pool_cleanup(retiring)
                self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 1)

        try:
            result = unittest.TestResult()
            FinalizerCleanup().run(result)
            self.assertEqual(result.failures, [])
            self.assertEqual(result.errors, [])
            self.assertIs(_pool_cleanup._PENDING, original)
            self.assertIn(followup, _pool_cleanup._snapshot())
        finally:
            followup.shutdown()

    def test_failed_fixture_cleanup_keeps_its_unfinished_root(self) -> None:
        """Restoring a fixture preserves failed cleanup obligations rather than hiding them."""
        original = _pool_cleanup._PENDING

        class PendingOwner:
            failed = True

            def shutdown(self) -> None:
                if self.failed:
                    raise TimeoutError("fixture cleanup remains pending")
                _pool_cleanup._release_pool_cleanup(self)

        owner = PendingOwner()

        class FailingCleanup(PoolCleanupTest):
            def runTest(self) -> None:
                _pool_cleanup._retain_pool_cleanup(owner)
                self.assertEqual(get_pending_pipeline_pool_cleanup_count(), 1)

        try:
            result = unittest.TestResult()
            FailingCleanup().run(result)
            self.assertEqual(result.failures, [])
            self.assertEqual(len(result.errors), 1)
            self.assertIn("fixture cleanup remains pending", result.errors[0][1])
            self.assertIs(_pool_cleanup._PENDING, original)
            self.assertIn(owner, _pool_cleanup._snapshot())
        finally:
            owner.failed = False
            owner.shutdown()


class WorkerRetirementTest(unittest.TestCase):
    def test_live_process_after_escalation_is_not_retired(self) -> None:
        """Even kill/join completion is not proof when exitcode remains unavailable."""
        proc = MagicMock(exitcode=None)
        with self.assertRaises(TimeoutError):
            _ProcessWorker(proc).stop()
        self.assertEqual(proc.join.call_count, 3)
        proc.terminate.assert_called_once_with()
        proc.kill.assert_called_once_with()

    def test_live_interpreter_is_not_closed(self) -> None:
        """A live interpreter thread prevents close and reports an incomplete retirement."""
        interp, thread = MagicMock(), MagicMock()
        thread.is_alive.return_value = True
        worker = _InterpreterWorker(interp, thread)
        with self.assertRaises(TimeoutError):
            worker.stop()
        interp.close.assert_not_called()
        thread.is_alive.return_value = False
        worker.stop()
        interp.close.assert_called_once_with()

    def test_interpreter_close_failure_propagates_for_retry(self) -> None:
        """A stopped thread is necessary but a failed interpreter close remains pending."""
        interp, thread = MagicMock(), MagicMock()
        thread.is_alive.return_value = False
        interp.close.side_effect = RuntimeError("close failed")
        worker = _InterpreterWorker(interp, thread)
        with self.assertLogs(
            "spdl.pipeline._subprocess_pipeline_pool", level="WARNING"
        ) as logs:
            with self.assertRaisesRegex(RuntimeError, "close failed"):
                worker.stop()
        self.assertIn("cleanup remains pending", logs.output[0])
        self.assertFalse(worker._closed)
        interp.close.side_effect = None
        worker.stop()
        self.assertTrue(worker._closed)
        worker.stop()
        self.assertEqual(interp.close.call_count, 2)
        self.assertEqual(thread.join.call_count, 2)
