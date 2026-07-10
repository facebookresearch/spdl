# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Main-process-owned worker pools that run a fused sub-pipeline inside each worker.

This is the streaming counterpart of :py:mod:`spdl.pipeline._subprocess_worker_pool`. Where that
module runs one submitted ``fn(*args)`` per task, here each worker process runs a nested
:py:class:`~spdl.pipeline.Pipeline` (built from the fused stages by
:py:func:`~spdl.pipeline._build.build_pipeline`). Items stream in over a queue and results stream
back over a shared queue, so the op→op handoff between fused stages stays inside one worker
process — no inter-stage IPC, and intermediate values need not be picklable.

Each worker has its own input queue in both layouts (chosen by ``continuous``):

- **Non-continuous:** the bridge round-robins items across the per-worker queues and sends one
  ``_SESSION_END`` to each; each worker runs one session (until its ``_SESSION_END``) and
  reports ``_DONE``. A per-worker queue (rather than one shared queue the workers steal from)
  guarantees every worker receives exactly one end marker, so a fast worker cannot steal a
  slow peer's marker and let the collector finish before the peer flushes its items.
- **Continuous:** the per-worker queue also lets the bridge broadcast epoch boundaries cleanly;
  each worker runs a long-lived *continuous* sub-pipeline, emitting ``_EPOCH_DONE`` at each
  boundary and staying warm across epochs.

Like :py:class:`spdl.pipeline._subprocess_worker_pool._WorkerPool`, the worker processes are
owned by the main process (spawned here, reaped in :py:meth:`_SubprocessPipelinePool.shutdown`).
The submit side is a small picklable :py:class:`_SubprocessPipelineHandle` carrying only the
queues, so the fused stage can drive the pool whether its node runs in the main process (a normal
``build``) or in a pipeline subprocess (``run_pipeline_in_subprocess``).

Messages are tagged with small integer kinds rather than sentinel objects: a sentinel pickled
onto a queue is a *different* object on the other side, so identity (``is``) comparison would
break across the process boundary.
"""

from __future__ import annotations

import logging
import queue as _queue
import traceback
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from typing import Any

from spdl.pipeline._components import (
    _DONE,
    _EPOCH,
    _EPOCH_DONE,
    _ERROR,
    _ITEM,
    _POOL_SHUTDOWN,
    _RESULT,
    _SESSION_END,
)
from spdl.pipeline.defs._defs import PipelineConfig, SourceConfig

__all__ = [
    "_InterpreterBackend",
    "_PoolBackend",
    "_ProcessBackend",
    "_SubprocessPipelineHandle",
    "_SubprocessPipelinePool",
    "_shutdown_pipeline_pools",
]

_LG: logging.Logger = logging.getLogger(__name__)


# Poll interval for a worker's blocking queue read, so a drain parked on an empty queue wakes to
# observe shutdown instead of blocking forever after the pool has been torn down.
_DRAIN_POLL_TIMEOUT: float = 0.5

# How long to wait for a subinterpreter worker thread to observe ``_POOL_SHUTDOWN`` and exit.
# Longer than the process backend's join bound because a subinterpreter cannot be force-killed --
# there is no ``terminate``/``kill`` escalation, so cooperative shutdown is the only lever.
_INTERPRETER_JOIN_TIMEOUT: float = 10.0

# Bounded blocking-put timeout for the shutdown marker on a subinterpreter worker's queue. See
# ``_InterpreterBackend.try_put_shutdown`` -- a subinterpreter worker cannot be force-killed, so
# try a little harder than a non-blocking put to deliver the marker, while still capping teardown.
_INTERPRETER_SHUTDOWN_PUT_TIMEOUT: float = 1.0


def _to_picklable_error(err: BaseException, tb: str) -> RuntimeError:
    """Reduce an arbitrary exception to a plainly-picklable error for the result queue.

    The original exception (e.g. a :py:class:`~spdl.pipeline.PipelineFailure` wrapping several
    sub-exceptions) may not be picklable, and ``mp.Queue.put`` pickles asynchronously — an
    unpicklable payload silently fails to arrive and hangs the consumer. So always relay a fresh
    ``RuntimeError`` carrying the text and the worker-side traceback (otherwise the traceback,
    the most useful part for debugging a fused run, is lost crossing the process boundary).
    """
    return RuntimeError(
        f"Fused subprocess pipeline failed: {type(err).__name__}: {err}\n{tb}"
    )


class _DrainSource:
    """Re-iterable source that drains one epoch of items from a worker's input queue.

    Used as a *continuous* source: :py:func:`spdl.pipeline._components._source._source_continuous`
    calls ``iter()`` once per epoch, so ``__iter__`` must return a fresh generator each time
    (a one-shot generator object would only ever run one epoch). Each pass yields ``_ITEM``
    payloads until an ``_EPOCH`` (epoch boundary) or ``_POOL_SHUTDOWN`` (teardown) message.

    ``exiting`` latches once ``_POOL_SHUTDOWN`` is seen; the worker loop reads it to stop after
    the current epoch. The blocking read uses a timeout so the drain thread wakes periodically
    to observe shutdown rather than parking forever on an empty queue during teardown.
    """

    def __init__(self, in_q: Any) -> None:
        self._in_q = in_q
        self.exiting = False

    def __iter__(self) -> Iterator[Any]:
        while True:
            try:
                kind, payload = self._in_q.get(timeout=_DRAIN_POLL_TIMEOUT)
            except _queue.Empty:
                if self.exiting:
                    return
                continue
            if kind == _ITEM:
                yield payload
            elif kind == _EPOCH:
                return
            else:  # _POOL_SHUTDOWN
                self.exiting = True
                return


def _run_sessions(
    in_q: Any,
    out_q: Any,
    sub_config: PipelineConfig[Any],
    build_kwargs: dict[str, Any],
) -> None:
    """Non-continuous worker body: one rebuilt sub-pipeline per input stream."""
    from spdl.pipeline._build import build_pipeline

    exiting = False
    session_ended = False

    def _drain() -> Iterator[Any]:
        nonlocal exiting, session_ended
        while True:
            kind, payload = in_q.get()
            if kind == _ITEM:
                yield payload
            elif kind == _SESSION_END:
                session_ended = True
                return
            else:  # _POOL_SHUTDOWN
                exiting = True
                return

    while not exiting:
        session_ended = False
        cfg = replace(sub_config, src=SourceConfig(_drain()))
        try:
            pipeline = build_pipeline(cfg, **build_kwargs)
            with pipeline.auto_stop():
                for out in pipeline:
                    out_q.put((_RESULT, out))
        except Exception as err:  # relayed to the submitter below
            out_q.put((_ERROR, _to_picklable_error(err, traceback.format_exc())))
            # A pipeline failure abandons the source generator mid-session, so this session's
            # remaining input (up to its ``_SESSION_END``) is still queued. Consume it here so
            # the session emits exactly one ``_DONE`` and stays aligned with the one-per-worker
            # ``_SESSION_END`` accounting on the feeder side — otherwise a retry session would
            # consume that marker and emit a second, stray ``_DONE``. ``KeyboardInterrupt`` /
            # ``SystemExit`` are not caught, so they tear the worker down as expected.
            while not session_ended and not exiting:
                kind, _ = in_q.get()
                if kind == _SESSION_END:
                    session_ended = True
                elif kind == _POOL_SHUTDOWN:
                    exiting = True
        out_q.put((_DONE, None))


def _run_continuous(
    in_q: Any,
    out_q: Any,
    sub_config: PipelineConfig[Any],
    build_kwargs: dict[str, Any],
) -> None:
    """Continuous worker body: one warm sub-pipeline, one ``_EPOCH_DONE`` per epoch.

    The sub-pipeline is built once with a continuous source draining ``in_q``; each
    ``for out in pipeline`` pass yields exactly one epoch's results (the continuous pipeline
    turns each ``_EPOCH_END`` into an iterator stop), after which the worker reports
    ``_EPOCH_DONE`` and loops for the next epoch — keeping the pipeline's prefetch buffers warm.

    Unlike :py:func:`_run_sessions`, a fatal sub-pipeline error is terminal here: the worker
    relays ``_ERROR`` + ``_DONE`` and exits rather than rebuilding for a retry. A continuous
    pipeline has no session boundary to recover at, and the bridge collector fails the whole
    pipeline (and tears the pool down) on the first ``_ERROR`` anyway, so there is no warm pool
    worth preserving.
    """
    from spdl.pipeline._build import build_pipeline

    source = _DrainSource(in_q)
    cfg = replace(sub_config, src=SourceConfig(source, continuous=True))
    try:
        pipeline = build_pipeline(cfg, **build_kwargs)
        with pipeline.auto_stop():
            while not source.exiting:
                for out in pipeline:
                    out_q.put((_RESULT, out))
                if source.exiting:
                    break
                out_q.put((_EPOCH_DONE, None))
    except Exception as err:
        # Catch ``Exception`` (not ``BaseException``) so ``KeyboardInterrupt`` / ``SystemExit``
        # propagate and actually tear the worker down, rather than being relayed as a
        # ``RuntimeError``. Matches :py:func:`_run_sessions` and the worker-loop initializer path.
        out_q.put((_ERROR, _to_picklable_error(err, traceback.format_exc())))
    out_q.put((_DONE, None))


def _pipeline_worker_loop(
    in_q: Any,
    out_q: Any,
    sub_config: PipelineConfig[Any],
    build_kwargs: dict[str, Any],
    continuous: bool,
    initializer: Callable[..., object] | None,
    initargs: tuple[Any, ...],
) -> None:
    """Worker entry point: run the initializer, then dispatch to the matching body."""
    if initializer is not None:
        try:
            initializer(*initargs)
        except Exception as err:
            # A failed initializer leaves this worker unable to run any session. Relay the error
            # (followed by a ``_DONE`` so the bridge collector's per-worker accounting stays
            # balanced) instead of exiting silently — a silent exit would hang the collector
            # waiting for messages this dead worker can never send. ``KeyboardInterrupt`` /
            # ``SystemExit`` deliberately propagate so the worker actually exits.
            out_q.put((_ERROR, _to_picklable_error(err, traceback.format_exc())))
            out_q.put((_DONE, None))
            return
    if continuous:
        _run_continuous(in_q, out_q, sub_config, build_kwargs)
    else:
        _run_sessions(in_q, out_q, sub_config, build_kwargs)


class _SubprocessPipelineHandle:
    """Picklable submit side of a :py:class:`_SubprocessPipelinePool`.

    Holds only the queues, worker count, and mode, so it is cheap to carry inside a
    :py:class:`~spdl.pipeline.defs.PipelineConfig` — including across the pickle boundary into a
    pipeline subprocess (the queues travel as ``Process`` spawn arguments, the only context in
    which an ``mp.Queue`` may be pickled). ``in_qs`` holds one input queue per worker in both
    modes. The worker processes themselves are owned by the main process, so this handle
    holds no process references.
    """

    def __init__(
        self, in_qs: list[Any], out_q: Any, max_workers: int, continuous: bool
    ) -> None:
        self.in_qs = in_qs
        self.out_q = out_q
        self.max_workers = max_workers
        self.continuous = continuous


class _Worker:
    """A spawned worker (process or subinterpreter thread) running the sub-pipeline loop."""

    def stop(self) -> None:
        """Reap the worker. Called after a ``_POOL_SHUTDOWN`` marker has been broadcast."""
        raise NotImplementedError


class _PoolBackend:
    """Process- vs subinterpreter-specific operations for :py:class:`_SubprocessPipelinePool`.

    The streaming protocol, message kinds, and worker body (:py:func:`_pipeline_worker_loop`)
    are identical across backends; only queue creation, worker spawning, and teardown differ.
    """

    def make_queue(self, maxsize: int) -> Any:
        raise NotImplementedError

    def spawn(self, target: Callable[..., object], args: tuple[Any, ...]) -> _Worker:
        raise NotImplementedError

    def try_put_shutdown(self, q: Any) -> None:
        """Best-effort put of a ``_POOL_SHUTDOWN`` marker onto ``q``.

        Must not block indefinitely: an implementation either puts without blocking or uses a
        bounded timeout, so teardown cannot wedge on a worker that has stopped consuming ``q``.
        """
        raise NotImplementedError

    def close_queue(self, q: Any) -> None:
        """Release this process's handle to ``q`` (a no-op where not applicable)."""
        raise NotImplementedError


class _ProcessWorker(_Worker):
    def __init__(self, proc: Any) -> None:
        self._proc = proc

    def stop(self) -> None:
        p = self._proc
        p.join(3)
        if p.exitcode is None:
            p.terminate()
            p.join(5)
        if p.exitcode is None:
            p.kill()
            p.join(5)


class _ProcessBackend(_PoolBackend):
    """Runs each worker in a separate OS process via a ``multiprocessing`` context."""

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx

    def make_queue(self, maxsize: int) -> Any:
        return self._ctx.Queue(maxsize=maxsize)

    def spawn(self, target: Callable[..., object], args: tuple[Any, ...]) -> _Worker:
        proc = self._ctx.Process(target=target, args=args, daemon=True)
        proc.start()
        return _ProcessWorker(proc)

    def try_put_shutdown(self, q: Any) -> None:
        # Non-blocking: the queues are bounded and on the error path workers may have stalled
        # on a full ``_out_q`` (the bridge stops draining after relaying an ``_ERROR``) and
        # stopped consuming, so a blocking ``put`` would wedge here forever and never reach the
        # terminate/kill escalation. A dropped marker is fine — the worker is reaped anyway.
        try:
            q.put_nowait((_POOL_SHUTDOWN, None))
        except Exception:
            # Expected when the queue is saturated (``queue.Full``) or otherwise unavailable;
            # the worker that misses the marker is reaped by ``stop`` regardless.
            pass

    def close_queue(self, q: Any) -> None:
        # Workers are already reaped, so items still buffered in this process's feeder threads
        # can never be delivered — ``cancel_join_thread`` first so ``join_thread`` does not
        # block forever flushing into a pipe no worker drains.
        q.cancel_join_thread()
        q.close()
        q.join_thread()


class _InterpreterWorker(_Worker):
    def __init__(self, interp: Any, thread: Any) -> None:
        self._interp = interp
        self._thread = thread

    def stop(self) -> None:
        # A subinterpreter thread cannot be force-killed; it exits when it observes the
        # ``_POOL_SHUTDOWN`` marker broadcast before ``stop``. Join with a bound so a wedged
        # worker cannot hang teardown forever.
        self._thread.join(timeout=_INTERPRETER_JOIN_TIMEOUT)
        if self._thread.is_alive():
            _LG.warning("Subinterpreter worker did not terminate gracefully.")
            return  # a still-running interpreter cannot be closed
        # Destroy the subinterpreter to release its resources; without this they accumulate
        # across repeated pipeline build/teardown cycles.
        try:
            self._interp.close()
        except Exception:
            _LG.warning("Failed to close subinterpreter worker.", exc_info=True)


class _InterpreterBackend(_PoolBackend):
    """Runs each worker in a Python subinterpreter via :py:mod:`concurrent.interpreters`.

    Requires Python 3.14+. Subinterpreters share the process (no ``mp_context``); an
    ``interpreters.Queue`` transports items by pickling, like an ``mp.Queue``.
    """

    def __init__(self) -> None:
        import concurrent.interpreters as interpreters  # pyre-ignore[21]

        # Typed ``Any``: ``concurrent.interpreters`` is Python 3.14+ only, so pyre (running under
        # an older config) cannot resolve its ``create``/``create_queue`` members.
        self._interpreters: Any = interpreters

    def make_queue(self, maxsize: int) -> Any:
        return self._interpreters.create_queue(maxsize=maxsize)

    def spawn(self, target: Callable[..., object], args: tuple[Any, ...]) -> _Worker:
        interp = self._interpreters.create()
        try:
            thread = interp.call_in_thread(target, *args)
        except BaseException:
            # ``call_in_thread`` failed after ``create()`` succeeded: close the orphaned
            # interpreter here, since outer cleanup only reaps workers this method returned.
            # Guard the close so a failure in it cannot mask the original error being re-raised.
            try:
                interp.close()
            except Exception:
                _LG.warning("Failed to close orphaned subinterpreter.", exc_info=True)
            raise
        return _InterpreterWorker(interp, thread)

    def try_put_shutdown(self, q: Any) -> None:
        # Best-effort but *bounded*: unlike the process backend, a subinterpreter worker that
        # misses this marker cannot be force-killed -- it would block on join and leak the
        # interpreter (its ``close()`` is skipped while still running). So try a little harder
        # than a non-blocking put: a short blocking put lets a briefly-full queue drain a slot.
        # The timeout still caps teardown latency and avoids wedging on a worker that has
        # stopped consuming (e.g. stalled on a full output queue after an error).
        try:
            q.put((_POOL_SHUTDOWN, None), timeout=_INTERPRETER_SHUTDOWN_PUT_TIMEOUT)
        except Exception:
            pass

    def close_queue(self, q: Any) -> None:
        # ``interpreters.Queue`` has no close(); nothing to release on this side.
        pass


class _SubprocessPipelinePool:
    """Main-process handle to a set of workers running a fused sub-pipeline.

    ``backend`` selects the worker kind (process or subinterpreter); everything else — the
    per-worker input queues, the shared output queue, and the shutdown/reap protocol — is
    identical across backends.
    """

    def __init__(
        self,
        backend: _PoolBackend,
        max_workers: int,
        sub_config: PipelineConfig[Any],
        build_kwargs: dict[str, Any],
        initializer: Callable[..., object] | None,
        initargs: tuple[Any, ...],
        continuous: bool = False,
    ) -> None:
        # Bound the queues so a slow consumer/producer applies backpressure rather than
        # buffering without limit. Sized to keep every worker fed, plus in-flight slack.
        size = max(4, max_workers * 2)
        self._backend = backend
        self._continuous = continuous
        self._out_q: Any = backend.make_queue(size)
        # One input queue per worker in both modes. Continuous mode needs it to broadcast epoch
        # boundaries cleanly; non-continuous mode needs it so each worker receives exactly one
        # ``_SESSION_END`` on its own queue. A single shared queue cannot guarantee that — a
        # worker that drains its stream and reaches ``_SESSION_END`` early can loop back and
        # steal a second marker meant for a slower peer still holding (un-flushed) items. That
        # peer then never ends, the collector reaches its ``_DONE`` count from the wrong workers
        # and finishes, and the slow peer's items are silently dropped. Per-worker queues remove
        # the shared marker entirely, so every worker ends exactly once.
        self._in_qs: list[Any] = [backend.make_queue(size) for _ in range(max_workers)]
        self._max_workers = max_workers
        started: list[_Worker] = []
        try:
            for i in range(max_workers):
                started.append(
                    backend.spawn(
                        _pipeline_worker_loop,
                        (
                            self._in_qs[i],
                            self._out_q,
                            sub_config,
                            build_kwargs,
                            continuous,
                            initializer,
                            initargs,
                        ),
                    )
                )
            self._workers: list[_Worker] = started
        except BaseException:
            # A ``spawn()`` partway through the loop (e.g. resource exhaustion) leaves the
            # already-started workers running with no owner: this half-constructed pool is
            # discarded by the caller and never reaches ``shutdown``. Broadcast the shutdown
            # marker first (as ``shutdown`` does): a subinterpreter worker blocks on its input
            # queue and cannot be force-killed, so without the marker it never exits, its join
            # times out, and its interpreter is leaked. Then reap the started workers and close
            # the queues before propagating, so the failure does not leak workers or pipe fds.
            self._broadcast_shutdown()
            self._reap(started)
            for q in (*self._in_qs, self._out_q):
                backend.close_queue(q)
            raise

    @staticmethod
    def _reap(workers: list[_Worker]) -> None:
        """Reap the given workers (join, escalating to terminate/kill where the backend can).

        Each worker is reaped independently: a failure stopping one (e.g. a subinterpreter that
        will not close) must not prevent the rest from being reaped, or the remaining workers and
        their pipe fds would leak.
        """
        for w in workers:
            try:
                w.stop()
            except Exception:
                _LG.warning("Failed to reap a worker.", exc_info=True)

    def make_handle(self) -> _SubprocessPipelineHandle:
        """Create the picklable submit-side handle that rides in the pipeline config."""
        return _SubprocessPipelineHandle(
            self._in_qs, self._out_q, self._max_workers, self._continuous
        )

    def _broadcast_shutdown(self) -> None:
        # One shutdown marker per worker, onto that worker's own input queue. Sequential by
        # design: the process backend's put is non-blocking, and the subinterpreter backend's
        # is bounded (``_INTERPRETER_SHUTDOWN_PUT_TIMEOUT``). That bound is only ever hit in the
        # rare error-teardown case where a worker's queue is full and it has stopped consuming;
        # worst-case teardown is then O(N) * the small per-worker bound -- acceptable on the
        # teardown path and not worth a parallel-put thread pool.
        for q in self._in_qs:
            self._backend.try_put_shutdown(q)

    def shutdown(self) -> None:
        """Tell every worker to exit, then reap them."""
        self._broadcast_shutdown()
        self._reap(self._workers)
        # Release this process's queue handles so any feeder threads created on first ``put``
        # exit; otherwise a process that creates many pipelines leaks threads and pipe fds.
        for q in (*self._in_qs, self._out_q):
            self._backend.close_queue(q)


def _shutdown_pipeline_pools(pools: Sequence[_SubprocessPipelinePool]) -> None:
    for pool in pools:
        pool.shutdown()
