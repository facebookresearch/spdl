# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Main-process-owned worker pools for ``run_pipeline_in_subprocess``.

When a pipe stage uses a stdlib :py:class:`~concurrent.futures.ProcessPoolExecutor` and the
pipeline is moved to a subprocess, naively reconstructing the executor *inside* that
subprocess spawns its worker processes as grandchildren of the main process. If the pipeline
subprocess is force-killed, those workers are never told to stop and become orphans.

This module avoids the nesting. When :py:func:`_hoist_process_pools` finds a stdlib
``ProcessPoolExecutor`` in a config, it spawns the worker processes in the **main** process
(as children of main, siblings of the pipeline subprocess) and replaces the executor with a
:py:class:`_RemoteExecutor` that merely holds the shared input/output queues. The pipeline
subprocess submits work onto those queues; the main process owns the workers and reaps them at
teardown.

``ProcessPoolExecutor``'s own design couples "submits work" with "owns workers" in a single
process, which is exactly what forces the grandchild nesting. By splitting those roles across
a small purpose-built executor we keep ownership in the main process without reaching into
CPython executor internals.
"""

from __future__ import annotations

import errno
import itertools
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
import warnings
import weakref
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import (
    BrokenExecutor,
    Executor,
    Future,
    InvalidStateError,
    ProcessPoolExecutor,
)
from multiprocessing.connection import wait as wait_for_mp_handles
from multiprocessing.reduction import ForkingPickler
from multiprocessing.util import register_after_fork
from typing import Any, NamedTuple, TypeVar

from spdl.pipeline._executor_proxy import (
    _ensure_executor_unused,
    _rewrite_config_executors,
)
from spdl.pipeline.defs import PipelineConfig

__all__ = [
    "_hoist_process_pools",
    "_IterableWithPoolShutdown",
    "_start_pool_monitors",
    "_shutdown_pools",
]

_T = TypeVar("_T")
_LG: logging.Logger = logging.getLogger(__name__)

_RESULT_QUEUE_RETRY_INITIAL_BACKOFF: float = 0.001
_RESULT_QUEUE_RETRY_MAX_BACKOFF: float = 0.05
_RESULT_QUEUE_FULL_MAX_ATTEMPTS: int = 8
_RESULT_QUEUE_TRANSIENT_MAX_ATTEMPTS: int = 8
_RESULT_QUEUE_RETRY_ERRNOS: tuple[int, ...] = (
    errno.EINTR,
    errno.EAGAIN,
    errno.EWOULDBLOCK,
)

# Sentinel placed on the input queue (one per worker) and output queue (one per router)
# to request shutdown.
_SHUTDOWN = None

_POOL_SHUTDOWN_REASON = "The worker pool shut down before the result was received."
_MONITOR_JOIN_TIMEOUT = 5.0

# A graceful output queue normally has only the small router sentinel left in its
# local feeder. Bound that feeder join so a dead router behind a full pipe cannot
# turn cleanup into an indefinite wait.
_OUTPUT_FEEDER_JOIN_TIMEOUT: float = 5.0


class _QueueSerializationBaseException(Exception):
    """An ``Exception`` wrapper that keeps direct ``BaseException`` failures recoverable."""

    def __init__(self, error: BaseException) -> None:
        self.error = error


def _load_queue_payload(data: bytes) -> Any:
    """Unwrap one asynchronously serialized queue payload."""
    return ForkingPickler.loads(data)


def _reduce_queue_payload(
    payload: Any,
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Serialize a payload once while normalizing direct ``BaseException`` failures.

    ``multiprocessing.Queue`` catches only ``Exception`` in its feeder thread. A user reducer
    that raises ``KeyboardInterrupt``, ``SystemExit``, or another direct ``BaseException``
    would otherwise kill that sole feeder and strand this and every later item. Framing the
    wire tuple inside the envelope lets us convert only that exceptional path to an
    ``Exception`` the queue's existing error hook can recover, without serializing user
    objects on the submitting thread or invoking their reducers twice.
    """
    try:
        data = bytes(ForkingPickler.dumps(payload))
    except Exception:
        raise
    except BaseException as error:  # noqa: B036 - normalize it for Queue's feeder hook
        raise _QueueSerializationBaseException(error) from None
    return _load_queue_payload, (data,)


class _WorkerStatus(NamedTuple):
    """Owner-to-submit-side notification for worker-pool liveness."""

    failure_reason: str | None


class _Submission:
    """Queue payload that reports a feeder-thread pickling failure to its Future."""

    def __init__(
        self,
        task_id: int,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        on_error: Callable[[int, BaseException], None],
    ) -> None:
        self._task_id = task_id
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        # Local-only state: ``__reduce__`` deliberately omits this bound callback.
        self._on_error = on_error

    def __reduce__(self) -> tuple[Callable[..., Any], tuple[Any, ...]]:
        # Successful queue serialization reconstructs the wire-compatible task tuple. If
        # serializing any user field fails, multiprocessing.Queue passes this original object
        # to its feeder-error hook, which can still reach the local Future callback.
        return _reduce_queue_payload(
            (self._task_id, self._fn, self._args, self._kwargs)
        )

    def _on_queue_feeder_error(self, err: BaseException) -> None:
        self._on_error(self._task_id, err)


def _serialization_error(label: str, err: BaseException) -> RuntimeError:
    """Create a plainly-picklable description of a serialization failure."""
    try:
        detail = str(err)
    except BaseException:  # noqa: B036 - an exception may have a broken ``__str__``
        detail = "<error message unavailable>"
    return RuntimeError(
        f"{label} could not be serialized: {type(err).__name__}: {detail}"
    )


class _Result:
    """Queue payload that replaces an unpicklable result with a safe error response."""

    def __init__(self, out_q: Any, task_id: int, ok: bool, payload: Any) -> None:
        self._out_q = out_q
        self._task_id = task_id
        self._ok = ok
        self._payload = payload

    def __reduce__(self) -> tuple[Callable[..., Any], tuple[Any, ...]]:
        # ``out_q`` is producer-local recovery state and must not cross the process boundary.
        return _reduce_queue_payload((self._task_id, self._ok, self._payload))

    def _on_queue_feeder_error(self, err: BaseException) -> None:
        label = "Worker result" if self._ok else "Worker exception"
        # This fallback contains only built-in, plainly-picklable values. Keep it a plain tuple
        # so a catastrophic failure serializing the fallback cannot recursively enqueue more
        # fallbacks through this handler.
        if getattr(self._out_q, "_closed", False) is True:
            return
        fallback = (self._task_id, False, _serialization_error(label, err))
        full_attempts = 0
        transport_attempts = 0
        full_backoff = _RESULT_QUEUE_RETRY_INITIAL_BACKOFF
        transport_backoff = _RESULT_QUEUE_RETRY_INITIAL_BACKOFF
        while True:
            try:
                # Never let the sole feeder thread block behind result backpressure. A
                # non-blocking put makes ``queue.Full`` observable so the bounded retry
                # policy below can either recover or terminate the worker deliberately.
                self._out_q.put(fallback, block=False)
            except BaseException as error:  # noqa: B036 - preserve worker liveness
                # Queue.put raises when local shutdown closes the queue. Re-check after
                # the call as close() can race this feeder-error recovery path.
                if getattr(self._out_q, "_closed", False) is True:
                    return
                if isinstance(error, queue.Full):
                    full_attempts += 1
                    if full_attempts < _RESULT_QUEUE_FULL_MAX_ATTEMPTS:
                        time.sleep(full_backoff)
                        full_backoff = min(
                            full_backoff * 2, _RESULT_QUEUE_RETRY_MAX_BACKOFF
                        )
                        continue
                if (
                    isinstance(error, OSError)
                    and error.errno in _RESULT_QUEUE_RETRY_ERRNOS
                ):
                    transport_attempts += 1
                    if transport_attempts < _RESULT_QUEUE_TRANSIENT_MAX_ATTEMPTS:
                        time.sleep(transport_backoff)
                        transport_backoff = min(
                            transport_backoff * 2,
                            _RESULT_QUEUE_RETRY_MAX_BACKOFF,
                        )
                        continue
                # The Future lives in another process, so this feeder thread cannot
                # resolve it directly. Make persistent or non-transient transport
                # failure observable through worker-process liveness instead of
                # silently stranding the Future.
                traceback.print_exc()
                os._exit(1)
                return
            else:
                return


def _handle_queue_feeder_error(err: BaseException, obj: Any) -> None:
    """Route an asynchronous ``mp.Queue`` feeder failure back to its submitter."""
    if isinstance(err, _QueueSerializationBaseException):
        err = err.error
    if isinstance(obj, (_Submission, _Result)):
        try:
            obj._on_queue_feeder_error(err)
        except BaseException:  # noqa: B036 - never kill the sole queue feeder silently
            traceback.print_exc()
        return
    # Once serialization succeeds, Queue replaces ``obj`` with serialized bytes before the
    # transport write. Such I/O failures cannot be recovered through the original envelope.
    # Terminate this worker so the owner observes the broken process and fails pending work;
    # merely returning would leave its Futures pending forever after the result was dropped.
    traceback.print_exception(type(err), err, err.__traceback__)
    os._exit(1)
    return


def _install_queue_feeder_error_handler(
    q: Any,
    handler: Callable[[BaseException, Any], None] = _handle_queue_feeder_error,
) -> None:
    """Install the hook CPython exposes for executor-specific feeder recovery."""
    # This private hook is intentionally provided "for overriding by concurrent.futures" and
    # is how ProcessPoolExecutor reports call-item pickling errors. It is present throughout
    # SPDL's supported CPython range (3.10+). Install it in each producer process before that
    # process starts its local feeder thread; Queue does not preserve arbitrary attributes when
    # it is pickled into another process.
    q._on_queue_feeder_error = handler


def _worker_loop(
    in_q: Any,
    out_q: Any,
    initializer: Callable[..., object] | None,
    initargs: tuple[Any, ...],
) -> None:
    """Worker body: run tasks pulled from ``in_q`` and push results onto ``out_q``.

    Each task is ``(task_id, fn, args, kwargs)``; each result is ``(task_id, ok, payload)``,
    where ``payload`` is the return value when ``ok`` else the raised exception. The producer
    wraps each tuple in an envelope whose reducer reconstructs this wire shape and whose
    feeder-error handler reports asynchronous pickling failures to the submitter.

    If the ``initializer`` raises, the worker keeps draining ``in_q`` and fails every routed
    task with a fresh, plainly-picklable error derived from it (rather than exiting and
    leaving the submitters' futures to hang forever on tasks that never get a response).
    """
    # ``mp.Queue.put`` serializes on a background feeder thread. Install the hook before this
    # process's first result so an unpicklable value is replaced instead of silently dropped.
    _install_queue_feeder_error_handler(out_q)
    init_error: BaseException | None = None
    if initializer is not None:
        try:
            initializer(*initargs)
        except BaseException as e:  # noqa: B036 - relayed to every submitter below
            init_error = e
    while True:
        task = in_q.get()
        if task is _SHUTDOWN:
            return
        task_id, fn, args, kwargs = task
        if init_error is not None:
            # Fail each task with a fresh error rather than re-sending the original exception
            # instance. Sharing one instance aliases its traceback and context across every
            # future it is set on; ``_Result`` also converts an unpicklable error to a plainly
            # picklable serialization failure.
            out_q.put(
                _Result(
                    out_q,
                    task_id,
                    False,
                    RuntimeError(f"Worker pool initializer failed: {init_error!r}"),
                )
            )
            continue
        try:
            result = fn(*args, **kwargs)
        except BaseException as e:  # noqa: B036 - relay any failure to the submitter
            out_q.put(_Result(out_q, task_id, False, e))
        else:
            out_q.put(_Result(out_q, task_id, True, result))


class _RemoteExecutor(Executor):
    """Submit side of a main-owned worker pool, designed to live in the pipeline subprocess.

    Holds only the shared input/output queues and a liveness channel (no worker handles), so it
    is cheap to pickle into the subprocess. On first :py:meth:`submit` it starts daemon
    threads that route results and observe worker-pool failure, resolving the matching
    :py:class:`~concurrent.futures.Future` in either case.

    It exposes ``_pool_executor_class = ProcessPoolExecutor`` so SPDL's ``_is_process_pool``
    detection treats it like a process pool (correct sync-generator batching and traceback
    wrapping). The workers themselves are owned and reaped by the main process, so
    :py:meth:`shutdown` here is intentionally a no-op.
    """

    _pool_executor_class: type[ProcessPoolExecutor] = ProcessPoolExecutor

    def __init__(
        self,
        in_q: Any,
        out_q: Any,
        max_workers: int,
        worker_status: Any,
        worker_status_close_lock: Any | None = None,
        worker_status_watcher_started: threading.Event | None = None,
    ) -> None:
        self._in_q = in_q
        self._out_q = out_q
        self._worker_status = worker_status
        # Direct same-process users share the owner's exact receive endpoint. Coordinate its
        # close with owner-side send/close operations; a subprocess receives a duplicated
        # endpoint and reconstructs this field as ``None``.
        self._worker_status_close_lock = worker_status_close_lock
        self._worker_status_watcher_started = worker_status_watcher_started
        self._worker_status_closed = False
        # Mirror ``ProcessPoolExecutor._max_workers`` so consumers that introspect a process
        # pool (e.g. pipeline-stats logging) can read the worker count off this proxy too —
        # it advertises ``_pool_executor_class = ProcessPoolExecutor``, so they expect it.
        self._max_workers = max_workers
        self._counter: itertools.count[int] = itertools.count()
        self._lock = threading.Lock()
        self._futures: dict[int, Future[Any]] = {}
        self._thread: threading.Thread | None = None
        self._worker_watcher: threading.Thread | None = None
        # Set (under ``_lock``) when a required helper cannot start or the router exits: those
        # helpers never restart, so further ``submit`` calls must fail fast rather than hang.
        self._broken: str | None = None
        _install_queue_feeder_error_handler(
            self._in_q, self._handle_input_queue_feeder_error
        )

    def _handle_input_queue_feeder_error(self, err: BaseException, obj: Any) -> None:
        """Fail submissions after an input transport drops serialized bytes."""
        if isinstance(obj, _Submission):
            _handle_queue_feeder_error(err, obj)
            return
        # Queue replaces the envelope with serialized bytes before writing to its
        # pipe. A failed write may have emitted a partial frame, so replaying the
        # bytes could corrupt the stream. The task id is also unavailable at this
        # point. Fail every pending submission and reject later ones rather than
        # leaving any Future unresolved or attempting to repair queue internals.
        traceback.print_exception(type(err), err, err.__traceback__)
        try:
            self._fail_pending(
                "Worker pool input queue transport failed after serialization."
            )
        except BaseException:  # noqa: B036 - never kill the sole queue feeder
            traceback.print_exc()

    @staticmethod
    def _run_after_helper_startup(
        startup_ready: threading.Event,
        startup_failed: threading.Event,
        target: Callable[[], None],
    ) -> None:
        """Run a helper only after every required thread starts successfully."""
        startup_ready.wait()
        if not startup_failed.is_set():
            target()

    def _ensure_router(self) -> None:
        # Double-checked locking so concurrent ``submit`` calls start exactly one router; two
        # routers would race on ``_out_q`` and each could claim results destined for the other,
        # silently dropping them and leaving callers blocked on ``Future.result()``.
        if self._thread is not None:
            return
        startup_failure_reason: str | None = None
        startup_ready = threading.Event()
        startup_failed = threading.Event()
        started_helpers: list[threading.Thread] = []
        try:
            with self._lock:
                if self._broken is not None:
                    self._close_worker_status()
                    raise BrokenExecutor(self._broken)
                if self._thread is not None:
                    return
                router = threading.Thread(
                    target=self._run_after_helper_startup,
                    args=(startup_ready, startup_failed, self._route),
                    name="spdl_remote_executor_router",
                    daemon=True,
                )
                worker_watcher = threading.Thread(
                    target=self._run_after_helper_startup,
                    args=(startup_ready, startup_failed, self._watch_worker_pool),
                    name="spdl_remote_executor_worker_watcher",
                    daemon=True,
                )
                try:
                    router.start()
                    started_helpers.append(router)
                    self._start_worker_watcher(worker_watcher)
                    started_helpers.append(worker_watcher)
                    self._thread = router
                    self._worker_watcher = worker_watcher
                except BaseException as e:  # noqa: B036 - permanently break partial startup
                    startup_failure_reason = (
                        f"Remote executor helper thread failed to start: {e!r}"
                    )
                    self._broken = startup_failure_reason
                    startup_failed.set()
                    self._close_worker_status()
                    raise
                finally:
                    # Never strand a helper behind the startup barrier, including
                    # when process control flow interrupts post-start bookkeeping.
                    startup_ready.set()
        except BaseException:
            if startup_failure_reason is not None:
                for helper in started_helpers:
                    helper.join()
                # Resolve callbacks only after releasing the non-reentrant executor lock.
                self._fail_pending(startup_failure_reason)
            raise

    def _start_worker_watcher(self, worker_watcher: threading.Thread) -> None:
        """Start the watcher and atomically transfer a same-process endpoint."""
        lock = self._worker_status_close_lock
        if lock is None:
            worker_watcher.start()
            return
        with lock:
            worker_watcher.start()
            if self._worker_status_watcher_started is not None:
                self._worker_status_watcher_started.set()

    def _close_worker_status(self) -> None:
        """Close the receive endpoint, coordinating exact same-process handles."""
        lock = self._worker_status_close_lock
        if lock is None:
            self._close_worker_status_unlocked()
            return
        with lock:
            self._close_worker_status_unlocked()

    def _close_worker_status_unlocked(self) -> None:
        if self._worker_status_closed:
            return
        # Latch before close so even an invalid endpoint is attempted and logged only once.
        self._worker_status_closed = True
        try:
            self._worker_status.close()
        except (EOFError, OSError, ValueError):
            _LG.exception("Failed to close worker pool status channel.")

    def _watch_worker_pool(self) -> None:
        """Fail pending work when the owner reports that the worker pool stopped."""
        try:
            try:
                status = self._worker_status.recv()
                failure_reason = status.failure_reason
            except (EOFError, OSError):
                failure_reason = "The worker pool status channel closed unexpectedly."
            except BaseException as e:  # noqa: B036 - fail work if this watcher breaks
                # Deserialization and malformed-message failures would otherwise kill this
                # non-restarting daemon thread and leave every pending Future unresolved.
                _LG.exception("Worker pool status watcher failed.")
                try:
                    detail = str(e)
                except BaseException:  # noqa: B036 - error may have broken __str__
                    detail = "<error message unavailable>"
                failure_reason = (
                    f"Worker pool status watcher failed: {type(e).__name__}: {detail}"
                )
        finally:
            # Closing a shared or already-invalid endpoint must not skip the
            # pending-Future failure below.
            self._close_worker_status()
        if failure_reason is not None:
            self._fail_pending(failure_reason)

    def _route(self) -> None:
        while True:
            try:
                result = self._out_q.get()
                if result is _SHUTDOWN:
                    self._fail_pending(_POOL_SHUTDOWN_REASON)
                    return
                task_id, ok, payload = result
            except (EOFError, OSError):
                # The queue was closed (e.g. teardown) before every result arrived. The router
                # is the sole consumer of ``_out_q`` and never restarts, so mark the executor
                # broken (failing the still-pending futures and any later ``submit``) instead of
                # leaving callers to hang forever on results that can no longer come back.
                self._fail_pending(
                    "Worker pool output queue closed before the result was received."
                )
                return
            except BaseException as e:  # noqa: B036 - relayed to every pending future below
                # Any other failure reading a result (e.g. ``get`` raising while unpickling a
                # malformed payload) would otherwise kill this sole, non-restarting router
                # thread silently and hang every pending future. Fail them fast with the cause.
                self._fail_pending(f"Worker pool result router failed: {e!r}")
                return
            with self._lock:
                fut = self._futures.pop(task_id, None)
            if fut is None:
                continue
            # The caller may have cancelled (or otherwise resolved) the future between
            # ``submit`` and now. Guard against ``InvalidStateError`` so one such future does
            # not kill the router thread and strand every later submission unresolved.
            try:
                if ok:
                    fut.set_result(payload)
                else:
                    fut.set_exception(payload)
            except InvalidStateError:
                pass

    def _fail_pending(self, reason: str) -> None:
        # Flip ``_broken`` and snapshot the pending futures under the same lock that ``submit``
        # uses to register, so there is no window where a future is enqueued after the snapshot
        # yet before ``_broken`` is visible — every future is either failed here or rejected by
        # ``submit``'s ``_broken`` check.
        with self._lock:
            if self._broken is None:
                self._broken = reason
            broken = self._broken
            pending = list(self._futures.values())
            self._futures.clear()
        control_flow_error: BaseException | None = None
        for fut in pending:
            if fut.done():
                continue
            try:
                fut.set_exception(BrokenExecutor(broken))
            except InvalidStateError:
                # The caller may have cancelled a Future after the snapshot. One cancelled
                # item must not prevent the remaining pending work from being failed.
                pass
            except (KeyboardInterrupt, SystemExit) as error:
                # Complete fanout before propagating process/thread control flow. Otherwise
                # one callback can strand later futures even though their map was cleared.
                if control_flow_error is None:
                    control_flow_error = error
            except BaseException:  # noqa: B036 - never strand later pending futures
                # Future invokes callbacks synchronously from set_exception(). A callback
                # failure outside Exception must not abort this loop and leave later work
                # unresolved after the futures map has already been cleared.
                _LG.warning(
                    "Pending-future callback raised during failure fanout.",
                    exc_info=True,
                )
        if control_flow_error is not None:
            raise control_flow_error

    def _fail_submission(self, task_id: int, err: BaseException) -> None:
        """Resolve one task whose request failed in the queue feeder thread."""
        with self._lock:
            fut = self._futures.pop(task_id, None)
        if fut is not None and not fut.done():
            try:
                fut.set_exception(err)
            except InvalidStateError:
                pass
            except BaseException:  # noqa: B036 - never kill the sole queue feeder
                traceback.print_exc()

    def submit(  # pyre-ignore[14]
        self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Future[Any]:
        self._ensure_router()
        fut: Future[Any] = Future()
        task_id = next(self._counter)
        with self._lock:
            if self._broken is not None:
                # The router has exited and will not restart; consuming ``_out_q`` is no longer
                # possible, so a future registered now would never resolve. Fail fast.
                raise BrokenExecutor(self._broken)
            self._futures[task_id] = fut
        try:
            self._in_q.put(
                _Submission(task_id, fn, args, kwargs, self._fail_submission)
            )
        except BaseException:
            # The task never reached the queue, so no result will ever come back for it. Drop
            # the registration so it does not linger unresolved, then surface the error.
            with self._lock:
                self._futures.pop(task_id, None)
            raise
        return fut

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        # The worker processes are owned by the main process and torn down there; the submit
        # side holds no worker handles, so there is nothing to shut down here.
        pass

    def __getstate__(self) -> dict[str, Any]:
        # Only IPC primitives + the worker count cross the pickle boundary; the threads and
        # pending futures are process-local and recreated lazily in the subprocess.
        return {
            "in_q": self._in_q,
            "out_q": self._out_q,
            "max_workers": self._max_workers,
            "worker_status": self._worker_status,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._in_q = state["in_q"]
        self._out_q = state["out_q"]
        self._max_workers = state["max_workers"]
        self._worker_status = state["worker_status"]
        self._worker_status_close_lock = None
        self._worker_status_watcher_started = None
        self._worker_status_closed = False
        self._counter = itertools.count()
        self._lock = threading.Lock()
        self._futures = {}
        self._thread = None
        self._worker_watcher = None
        self._broken = None
        _install_queue_feeder_error_handler(
            self._in_q, self._handle_input_queue_feeder_error
        )


def _close_connection_after_fork(connection: Any) -> None:
    """Close an owner-only pipe endpoint inherited by a forked child."""
    connection.close()


class _WorkerPool:
    """Main-process handle to a set of worker processes feeding a single queue pair."""

    def __init__(
        self,
        ctx: Any,
        max_workers: int,
        initializer: Callable[..., object] | None,
        initargs: tuple[Any, ...],
        defer_monitor: bool = False,
    ) -> None:
        self._in_q: Any = ctx.Queue()
        self._out_q: Any = ctx.Queue()
        self._max_workers = max_workers
        self._closed = False
        # The remote executor cannot safely infer worker death from ``out_q`` EOF because the
        # owner retains a writer. Even if that writer were closed, killing a worker midway
        # through a queue write could leave the result pipe or its write lock wedged. Publish
        # pool state through a pipe that never shares the result transport. The two Events are
        # owner-process-only flags: a multiprocessing.Event waiter that is killed can leave its
        # Condition bookkeeping inconsistent and deadlock a later Event.set() during teardown.
        self._shutdown_started = threading.Event()
        self._worker_failed = threading.Event()
        self._procs: list[Any] = [
            ctx.Process(
                target=_worker_loop,
                args=(self._in_q, self._out_q, initializer, initargs),
                daemon=True,
            )
            for _ in range(max_workers)
        ]
        started: list[Any] = []
        try:
            for p in self._procs:
                p.start()
                started.append(p)
        except BaseException:
            # A ``start()`` partway through the loop (e.g. resource exhaustion) leaves the
            # already-started workers running with no owner: this half-constructed pool is
            # discarded by the caller and never reaches ``_shutdown_pools``. Tear the started
            # workers down and close the queues before propagating, so the failure does not
            # leak processes or pipe fds.
            self._terminate(started)
            for q in (self._in_q, self._out_q):
                q.close()
                q.join_thread()
            raise
        worker_status_recv: Any | None = None
        worker_status_send: Any | None = None
        try:
            worker_status_recv, worker_status_send = ctx.Pipe(duplex=False)
            # The write endpoint belongs exclusively to the owner process. In particular, the
            # pipeline subprocess must not retain a copy: otherwise owner closure would not
            # deliver EOF to its watcher. Pool workers were started before the pipe was
            # created, so they cannot inherit either endpoint under ``fork``.
            register_after_fork(worker_status_send, _close_connection_after_fork)
        except BaseException:
            for connection in (worker_status_recv, worker_status_send):
                if connection is not None:
                    try:
                        connection.close()
                    except (EOFError, OSError):
                        pass
            self._terminate(self._procs)
            for q in (self._in_q, self._out_q):
                q.close()
                q.join_thread()
            raise
        self._worker_status_recv = worker_status_recv
        self._worker_status_send = worker_status_send
        self._status_receiver_transferred = False
        self._status_receiver_watcher_started = threading.Event()
        self._status_lock = threading.Lock()
        self._status_sent = False
        self._monitor: threading.Thread | None = None
        if not defer_monitor:
            try:
                self._start_monitor()
            except BaseException:
                self.shutdown()
                raise

    def _start_monitor(self) -> None:
        """Start the worker-liveness monitor after all forked pools are constructed."""
        if self._monitor is not None:
            return
        monitor = threading.Thread(
            target=self._monitor_workers,
            name="spdl_worker_pool_monitor",
            daemon=True,
        )
        monitor.start()
        self._monitor = monitor

    def _monitor_workers(self) -> None:
        """Break the remote executor when any worker exits before pool shutdown."""
        try:
            wait_for_mp_handles([proc.sentinel for proc in self._procs])
        except BaseException as error:  # noqa: B036 - this sole monitor must report failure
            if self._shutdown_started.is_set():
                return
            _LG.exception("Worker pool monitor failed.")
            try:
                detail = str(error)
            except BaseException:  # noqa: B036 - error may have broken __str__
                detail = "<error message unavailable>"
            failure_reason = (
                f"Worker pool monitor failed: {type(error).__name__}: {detail}"
            )
        else:
            failure_reason = "A worker process exited unexpectedly."
        if self._shutdown_started.is_set():
            return
        self._worker_failed.set()
        self._send_worker_status(failure_reason)

    def _send_worker_status(self, failure_reason: str | None) -> None:
        """Wake the remote watcher once without sharing a kill-sensitive semaphore."""
        with self._status_lock:
            if self._status_sent:
                return
            try:
                self._worker_status_send.send(_WorkerStatus(failure_reason))
            except (EOFError, OSError, ValueError):
                # A failed send cannot leave the receiver blocked forever. Close
                # this endpoint so the watcher observes EOF and fails pending work.
                try:
                    self._worker_status_send.close()
                except (EOFError, OSError, ValueError):
                    _LG.exception("Failed to close worker pool status sender.")
                finally:
                    self._status_sent = True
            else:
                self._status_sent = True

    def make_executor(self, *, receiver_is_shared: bool = True) -> _RemoteExecutor:
        """Create the submit-side executor that rides in the pipeline config."""
        # In the usual subprocess path the executor receives a duplicated handle,
        # so owner shutdown still closes its original. Direct same-process use
        # shares this exact Connection. Its watcher atomically takes cleanup
        # ownership when it starts; until then owner shutdown retains fallback
        # ownership of the unused endpoint.
        with self._status_lock:
            executor = _RemoteExecutor(
                self._in_q,
                self._out_q,
                self._max_workers,
                self._worker_status_recv,
                self._status_lock,
                (self._status_receiver_watcher_started if receiver_is_shared else None),
            )
            self._status_receiver_transferred = receiver_is_shared
        return executor

    @staticmethod
    def _close_queue(
        q: Any,
        *,
        abandon: bool,
        feeder_join_timeout: float | None = None,
    ) -> bool:
        """Close a queue and report whether its local feeder fully stopped."""
        feeder_stopped = not abandon
        if abandon:
            try:
                q.cancel_join_thread()
            except (EOFError, OSError, ValueError):
                feeder_stopped = False
                _LG.warning(
                    "Failed to cancel the worker-pool queue feeder join.",
                    exc_info=True,
                )
        try:
            q.close()
        except (EOFError, OSError, ValueError):
            feeder_stopped = False
            _LG.warning("Failed to close a worker-pool queue.", exc_info=True)

        if not abandon and feeder_join_timeout is not None:
            # ``_thread`` is a CPython multiprocessing.Queue implementation
            # detail. Without a real Thread there is no way to impose the
            # requested bound on a compatible queue's join_thread(), so abandon
            # its implicit join rather than risk blocking teardown indefinitely.
            feeder = getattr(q, "_thread", None)
            if isinstance(feeder, threading.Thread):
                feeder.join(feeder_join_timeout)
                if feeder.is_alive():
                    feeder_stopped = False
                    try:
                        q.cancel_join_thread()
                    except (EOFError, OSError, ValueError):
                        _LG.warning(
                            "Failed to cancel a stalled worker-pool queue feeder join.",
                            exc_info=True,
                        )
            else:
                feeder_stopped = False
                try:
                    q.cancel_join_thread()
                except (EOFError, OSError, ValueError):
                    _LG.warning(
                        "Failed to cancel an unbounded compatible queue feeder join.",
                        exc_info=True,
                    )
        if feeder_stopped:
            try:
                q.join_thread()
            except (EOFError, OSError, ValueError):
                feeder_stopped = False
                _LG.warning("Failed to join a worker-pool queue feeder.", exc_info=True)
        return feeder_stopped

    @staticmethod
    def _terminate(procs: list[Any]) -> bool:
        """Reap workers, returning whether any required forced termination."""
        forced = False
        for p in procs:
            p.join(3)
            if p.exitcode is None:
                forced = True
                p.terminate()
                p.join(5)
            if p.exitcode is None:
                p.kill()
                p.join(5)
        return forced

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._shutdown_started.set()
        for _ in self._procs:
            try:
                self._in_q.put(_SHUTDOWN)
            except Exception:
                # A failed sentinel for one worker should not prevent the others from
                # receiving theirs; fall through to the join/terminate/kill escalation below.
                continue
        forced = self._terminate(self._procs)
        unexpectedly_exited = any(
            proc.exitcode not in (None, 0) for proc in self._procs
        )
        if self._monitor is not None:
            # Usually _terminate() makes a worker sentinel ready. A process stuck
            # in an uninterruptible state may survive even the bounded kill/join
            # escalation, so never let its monitor wedge owner teardown forever.
            # Give the monitor a chance to publish a worker failure before feeder
            # policy is chosen. A delayed monitor alone does not make a cleanly
            # reaped worker unsafe to flush.
            self._monitor.join(timeout=_MONITOR_JOIN_TIMEOUT)
        abandon_feeders = forced or unexpectedly_exited or self._worker_failed.is_set()
        status_reason = _POOL_SHUTDOWN_REASON if abandon_feeders else None
        # Wake a result router that is blocked on an otherwise-idle output queue. Queue.close()
        # alone does not close the reader in a process that has never produced onto that queue,
        # so without an explicit sentinel direct users of ``_WorkerPool`` leak one daemon thread
        # per executor. Prefer a nonblocking enqueue, then allow one bounded wait
        # if a compatible queue reports that it is full. The feeder itself is also
        # joined with a bound below because its pipe write can still block after
        # this call returns.
        # Even forced worker teardown still has a live result router to wake.
        # Give a successfully enqueued output sentinel the bounded flush below;
        # only abandon this feeder if enqueueing or that bounded flush fails.
        abandon_output_feeder = False
        try:
            try:
                self._out_q.put_nowait(_SHUTDOWN)
            except queue.Full:
                self._out_q.put(
                    _SHUTDOWN,
                    timeout=_OUTPUT_FEEDER_JOIN_TIMEOUT,
                )
        except queue.Full:
            abandon_output_feeder = True
            status_reason = _POOL_SHUTDOWN_REASON
        except (EOFError, OSError, ValueError):
            abandon_output_feeder = True
            status_reason = _POOL_SHUTDOWN_REASON
            _LG.warning(
                "Failed to enqueue the worker-pool output shutdown marker.",
                exc_info=True,
            )
        except BaseException:
            status_reason = _POOL_SHUTDOWN_REASON
            raise
        finally:
            try:
                # Close this process's queue handles so the feeder thread started when the
                # main process put the shutdown sentinels exits; otherwise a long-lived
                # main process leaks feeder threads and pipe fds. A feeder can still be
                # blocked after its only reader exits: forced teardown abandons both
                # queues, while graceful teardown gives the output feeder only a bounded
                # opportunity to flush its router sentinel.
                try:
                    self._close_queue(self._in_q, abandon=abandon_feeders)
                finally:
                    self._close_queue(
                        self._out_q,
                        abandon=abandon_output_feeder,
                        feeder_join_timeout=_OUTPUT_FEEDER_JOIN_TIMEOUT,
                    )
            finally:
                # A bounded feeder join timing out does not prove the sentinel was
                # lost: a live result reader can still drain the pipe and let the
                # daemon feeder finish asynchronously. Report failure only for the
                # worker or enqueue failures identified above.
                try:
                    self._send_worker_status(status_reason)
                finally:
                    with self._status_lock:
                        if (
                            not self._status_receiver_transferred
                            or not self._status_receiver_watcher_started.is_set()
                        ):
                            self._worker_status_recv.close()
                        self._worker_status_send.close()
        # Release the queue-owned SemLocks promptly instead of retaining them on
        # this finalized pool until interpreter shutdown.
        self._in_q = None
        self._out_q = None


def _hoist_process_pools(
    config: PipelineConfig[Any],
    mp_context: str | None = None,
) -> tuple[PipelineConfig[Any], list[_WorkerPool]]:
    """Move stdlib ``ProcessPoolExecutor`` workers into the main process.

    Returns a rewritten config in which each stdlib
    :py:class:`~concurrent.futures.ProcessPoolExecutor` attached to a pipe is replaced with a
    :py:class:`_RemoteExecutor`, plus the list of :py:class:`_WorkerPool` handles that own the
    spawned workers (the caller must start their liveness monitors after any subsequent
    process creation, then :py:func:`_shutdown_pools` them at teardown).

    ``mp_context`` is the multiprocessing start-method name (as accepted by
    :py:func:`multiprocessing.get_context`). The context is created lazily, only when a
    ``ProcessPoolExecutor`` is actually found, so configs without one incur no cost.

    A single ``ProcessPoolExecutor`` instance reused across multiple pipes maps to one shared
    pool (and one shared ``_RemoteExecutor``), preserving the user's intent to share workers.
    Non-``ProcessPoolExecutor`` executors (``ThreadPoolExecutor``, SPDL ``Priority*`` pools,
    ``None``) are left untouched. The input ``config`` is not mutated.
    """
    pools: list[_WorkerPool] = []
    seen: dict[int, _RemoteExecutor] = {}
    ctx_box: list[Any] = []  # one-element cache for the lazily-created context

    def convert(executor: Any) -> Any:
        if type(executor) is not ProcessPoolExecutor:
            return executor
        key = id(executor)
        if key in seen:
            return seen[key]
        _ensure_executor_unused(executor)
        if not ctx_box:
            ctx = mp.get_context(mp_context)
            if ctx.get_start_method() == "fork" and threading.active_count() > 1:
                # Spawning the worker pool with ``fork`` from a multi-threaded process can
                # deadlock: ``fork`` copies only the calling thread, so a lock another thread
                # holds is never released in the child. Warn (not raise) — a single-threaded
                # caller is fine, and the user may knowingly accept the risk.
                warnings.warn(
                    "Hoisting a ProcessPoolExecutor for run_pipeline_in_subprocess with the "
                    "'fork' start method from a multi-threaded process can deadlock. Pass "
                    "mp_context='spawn' or 'forkserver', or build the pipeline before other "
                    "threads start.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            ctx_box.append(ctx)
        # ``executor`` is statically a ``ProcessPoolExecutor`` here, whose private worker
        # config attributes are not part of its declared type; read them off an ``Any`` alias.
        ppe: Any = executor
        pool = _WorkerPool(
            ctx_box[0],
            ppe._max_workers,
            ppe._initializer,
            ppe._initargs,
            defer_monitor=True,
        )
        pools.append(pool)
        remote = pool.make_executor(receiver_is_shared=False)
        seen[key] = remote
        return remote

    try:
        new_config = _rewrite_config_executors(config, convert)
    except BaseException:
        # If ``convert`` raises after earlier pools were already constructed (e.g. a later
        # ``_WorkerPool`` fails on OOM or fork/spawn failure), the partially-built ``pools``
        # never reach the caller, so reap them here before propagating instead of leaking
        # their worker processes and pipe fds.
        _shutdown_pools(pools)
        raise
    return new_config, pools


def _start_pool_monitors(pools: list[_WorkerPool]) -> None:
    """Start liveness monitors after every process that may use ``fork`` is spawned."""
    # A monitor is a live thread. Starting it before another pool or the outer pipeline
    # process is forked could copy a lock held by that thread into the child and deadlock it.
    # Pool construction therefore defers monitor startup until the caller has completed all
    # process creation; ``_start_monitor`` is idempotent for rollback-friendly callers.
    for pool in pools:
        pool._start_monitor()


def _shutdown_pools(pools: list[_WorkerPool]) -> None:
    for pool in pools:
        pool.shutdown()


def _teardown_inner_then_pools(
    inner: Iterable[object], pools: list[_WorkerPool]
) -> None:
    # Join the worker subprocess first (the iterable's own finalizer terminates and joins it),
    # then reap the pools it submits to. Reaping the pools while the subprocess is still live
    # could close the shared queues from under a mid-submit worker and produce broken-pipe
    # noise or dropped results. The ``_finalizer`` attribute is the iterable's documented
    # teardown handle; ``test_subprocess_iterable_exposes_finalizer`` guards against a rename
    # silently turning this into a no-op (which would skip the join and leave the hazard).
    if (inner_finalizer := getattr(inner, "_finalizer", None)) is not None:
        inner_finalizer()
    _shutdown_pools(pools)


class _IterableWithPoolShutdown(Iterable[_T]):
    """Wraps an iterable so hoisted worker pools are reaped once the wrapper is torn down.

    ``run_pipeline_in_subprocess`` returns an :py:class:`~collections.abc.Iterable`, so the
    concrete class can be swapped without changing the public contract. The hoisted
    :py:class:`~concurrent.futures.ProcessPoolExecutor` workers live in the main process and
    must outlive every iteration: the returned iterable is re-iterated once per epoch and the
    subprocess keeps submitting to the pools across epochs (a continuous source keeps the
    pipeline running between them). So the teardown is tied to this wrapper's finalizer rather
    than to :py:meth:`__iter__` — the pools persist across re-iterations and are reaped exactly
    once, when the wrapper is garbage-collected after the epoch loop (or eagerly, if the caller
    never iterates it). This keeps the pool lifetime out of the generic
    :py:func:`~spdl.pipeline._iter_utils.iterate_in_subprocess` API.

    Args:
        inner: The iterable returned by
            :py:func:`~spdl.pipeline._iter_utils.iterate_in_subprocess`.
        pools: The hoisted worker pools to shut down once the wrapper is torn down.
    """

    def __init__(self, inner: Iterable[_T], pools: list[_WorkerPool]) -> None:
        self._inner = inner
        self._pools = pools
        self._finalizer: weakref.finalize = weakref.finalize(
            self, _teardown_inner_then_pools, inner, pools
        )

    def __iter__(self) -> Iterator[_T]:
        return iter(self._inner)
