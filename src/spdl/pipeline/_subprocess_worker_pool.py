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
from multiprocessing.reduction import ForkingPickler
from typing import Any, TypeVar

from spdl.pipeline._executor_proxy import (
    _ensure_executor_unused,
    _rewrite_config_executors,
)
from spdl.pipeline.defs import PipelineConfig

__all__ = [
    "_hoist_process_pools",
    "_IterableWithPoolShutdown",
    "_shutdown_pools",
]

_T = TypeVar("_T")

_RESULT_QUEUE_RETRY_INITIAL_BACKOFF: float = 0.001
_RESULT_QUEUE_RETRY_MAX_BACKOFF: float = 0.05
_RESULT_QUEUE_FULL_MAX_ATTEMPTS: int = 8
_RESULT_QUEUE_TRANSIENT_MAX_ATTEMPTS: int = 8
_RESULT_QUEUE_RETRY_ERRNOS: tuple[int, ...] = (
    errno.EINTR,
    errno.EAGAIN,
    errno.EWOULDBLOCK,
)

# Sentinel placed on the input queue (one per worker) to request shutdown.
_SHUTDOWN = None


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

    Holds only the shared input/output queues (no worker handles), so it is cheap to pickle
    into the subprocess. On first :py:meth:`submit` it starts a daemon routing thread that
    reads results off ``out_q`` and resolves the matching
    :py:class:`~concurrent.futures.Future`.

    It exposes ``_pool_executor_class = ProcessPoolExecutor`` so SPDL's ``_is_process_pool``
    detection treats it like a process pool (correct sync-generator batching and traceback
    wrapping). The workers themselves are owned and reaped by the main process, so
    :py:meth:`shutdown` here is intentionally a no-op.
    """

    _pool_executor_class: type[ProcessPoolExecutor] = ProcessPoolExecutor

    def __init__(self, in_q: Any, out_q: Any, max_workers: int) -> None:
        self._in_q = in_q
        self._out_q = out_q
        # Mirror ``ProcessPoolExecutor._max_workers`` so consumers that introspect a process
        # pool (e.g. pipeline-stats logging) can read the worker count off this proxy too —
        # it advertises ``_pool_executor_class = ProcessPoolExecutor``, so they expect it.
        self._max_workers = max_workers
        self._counter: itertools.count[int] = itertools.count()
        self._lock = threading.Lock()
        self._futures: dict[int, Future[Any]] = {}
        self._thread: threading.Thread | None = None
        # Set (under ``_lock``) when the router thread exits because ``_out_q`` closed: the
        # router never restarts, so further ``submit`` calls must fail fast rather than hang.
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

    def _ensure_router(self) -> None:
        # Double-checked locking so concurrent ``submit`` calls start exactly one router; two
        # routers would race on ``_out_q`` and each could claim results destined for the other,
        # silently dropping them and leaving callers blocked on ``Future.result()``.
        if self._thread is None:
            with self._lock:
                if self._thread is None:
                    self._thread = threading.Thread(
                        target=self._route,
                        name="spdl_remote_executor_router",
                        daemon=True,
                    )
                    self._thread.start()

    def _route(self) -> None:
        while True:
            try:
                task_id, ok, payload = self._out_q.get()
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
            self._broken = reason
            pending = list(self._futures.values())
            self._futures.clear()
        for fut in pending:
            if not fut.done():
                try:
                    fut.set_exception(BrokenExecutor(reason))
                except InvalidStateError:
                    pass
                except BaseException:  # noqa: B036 - keep failing later Futures
                    traceback.print_exc()

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
        # Only the queues + worker count cross the pickle boundary; the router thread and
        # pending futures are process-local and recreated lazily in the subprocess.
        return {
            "in_q": self._in_q,
            "out_q": self._out_q,
            "max_workers": self._max_workers,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._in_q = state["in_q"]
        self._out_q = state["out_q"]
        self._max_workers = state["max_workers"]
        self._counter = itertools.count()
        self._lock = threading.Lock()
        self._futures = {}
        self._thread = None
        self._broken = None
        _install_queue_feeder_error_handler(
            self._in_q, self._handle_input_queue_feeder_error
        )


class _WorkerPool:
    """Main-process handle to a set of worker processes feeding a single queue pair."""

    def __init__(
        self,
        ctx: Any,
        max_workers: int,
        initializer: Callable[..., object] | None,
        initargs: tuple[Any, ...],
    ) -> None:
        self._in_q: Any = ctx.Queue()
        self._out_q: Any = ctx.Queue()
        self._max_workers = max_workers
        self._closed = False
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

    def make_executor(self) -> _RemoteExecutor:
        """Create the submit-side executor that rides in the pipeline config."""
        return _RemoteExecutor(self._in_q, self._out_q, self._max_workers)

    @staticmethod
    def _terminate(procs: list[Any]) -> None:
        """Join the given worker processes, escalating to terminate/kill if they don't exit."""
        for p in procs:
            p.join(3)
            if p.exitcode is None:
                p.terminate()
                p.join(5)
            if p.exitcode is None:
                p.kill()
                p.join(5)

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        for _ in self._procs:
            try:
                self._in_q.put(_SHUTDOWN)
            except Exception:
                # A failed sentinel for one worker should not prevent the others from
                # receiving theirs; fall through to the join/terminate/kill escalation below.
                continue
        self._terminate(self._procs)
        # Close this process's queue handles so the feeder thread started when the main
        # process put the shutdown sentinels exits; otherwise a long-lived main process that
        # creates and destroys many pipelines leaks feeder threads and pipe fds.
        for q in (self._in_q, self._out_q):
            q.close()
            q.join_thread()
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
    spawned workers (the caller must :py:func:`_shutdown_pools` them at teardown).

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
        )
        pools.append(pool)
        remote = pool.make_executor()
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
