# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

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

import itertools
import multiprocessing as mp
import threading
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

# Sentinel placed on the input queue (one per worker) to request shutdown.
_SHUTDOWN = None


def _worker_loop(
    in_q: Any,
    out_q: Any,
    initializer: Callable[..., object] | None,
    initargs: tuple[Any, ...],
) -> None:
    """Worker body: run tasks pulled from ``in_q`` and push results onto ``out_q``.

    Each task is ``(task_id, fn, args, kwargs)``; each result is ``(task_id, ok, payload)``
    where ``payload`` is the return value when ``ok`` else the raised exception. Exceptions
    must be picklable to travel back over ``out_q`` (the same constraint that already applies
    to everything crossing a process boundary in SPDL).

    If the ``initializer`` raises, the worker keeps draining ``in_q`` and fails every routed
    task with a fresh, plainly-picklable error derived from it (rather than exiting and
    leaving the submitters' futures to hang forever on tasks that never get a response).
    """
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
            # Fail each task with a fresh, plainly-picklable error rather than re-sending the
            # original exception instance: the original may not be picklable (``mp.Queue.put``
            # pickles asynchronously, so an unpicklable payload silently fails to arrive and
            # hangs the future), and sharing one instance aliases its traceback and context
            # across every future it is set on.
            out_q.put(
                (
                    task_id,
                    False,
                    RuntimeError(f"Worker pool initializer failed: {init_error!r}"),
                )
            )
            continue
        try:
            out_q.put((task_id, True, fn(*args, **kwargs)))
        except BaseException as e:  # noqa: B036 - relay any failure to the submitter
            out_q.put((task_id, False, e))


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
                fut.set_exception(BrokenExecutor(reason))

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
            self._in_q.put((task_id, fn, args, kwargs))
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
