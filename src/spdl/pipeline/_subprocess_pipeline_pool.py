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

Two layouts, chosen by ``continuous``:

- **Non-continuous:** all workers steal from one shared input queue; each worker runs one session
  (until ``_SESSION_END``) and reports ``_DONE``.
- **Continuous:** each worker has its own input queue (so the bridge can broadcast epoch
  boundaries) and runs a long-lived *continuous* sub-pipeline, emitting ``_EPOCH_DONE`` at each
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
    "_SubprocessPipelineHandle",
    "_SubprocessPipelinePool",
    "_shutdown_pipeline_pools",
]


# Poll interval for a worker's blocking queue read, so a drain parked on an empty queue wakes to
# observe shutdown instead of blocking forever after the pool has been torn down.
_DRAIN_POLL_TIMEOUT: float = 0.5


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
    which an ``mp.Queue`` may be pickled). ``in_qs`` is the shared input queue (length 1) in
    non-continuous mode, or one queue per worker in continuous mode. The worker processes
    themselves are owned by the main process, so this handle holds no process references.
    """

    def __init__(
        self, in_qs: list[Any], out_q: Any, max_workers: int, continuous: bool
    ) -> None:
        self.in_qs = in_qs
        self.out_q = out_q
        self.max_workers = max_workers
        self.continuous = continuous


class _SubprocessPipelinePool:
    """Main-process handle to a set of workers running a fused sub-pipeline."""

    def __init__(
        self,
        ctx: Any,
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
        self._continuous = continuous
        self._out_q: Any = ctx.Queue(maxsize=size)
        if continuous:
            # One queue per worker so the bridge can broadcast epoch boundaries cleanly.
            self._in_qs: list[Any] = [
                ctx.Queue(maxsize=size) for _ in range(max_workers)
            ]
            worker_in_qs = self._in_qs
        else:
            # One shared queue the workers steal from.
            shared = ctx.Queue(maxsize=size)
            self._in_qs = [shared]
            worker_in_qs = [shared] * max_workers
        self._max_workers = max_workers
        self._procs: list[Any] = [
            ctx.Process(
                target=_pipeline_worker_loop,
                args=(
                    worker_in_qs[i],
                    self._out_q,
                    sub_config,
                    build_kwargs,
                    continuous,
                    initializer,
                    initargs,
                ),
                daemon=True,
            )
            for i in range(max_workers)
        ]
        started: list[Any] = []
        try:
            for p in self._procs:
                p.start()
                started.append(p)
        except BaseException:
            # A ``start()`` partway through the loop (e.g. resource exhaustion) leaves the
            # already-started workers running with no owner: this half-constructed pool is
            # discarded by the caller and never reaches ``shutdown``. Tear the started workers
            # down and close the queues before propagating, so the failure does not leak
            # processes or pipe fds. Mirrors ``_WorkerPool.__init__``.
            self._terminate(started)
            for q in (*self._in_qs, self._out_q):
                q.close()
                q.join_thread()
            raise

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

    def make_handle(self) -> _SubprocessPipelineHandle:
        """Create the picklable submit-side handle that rides in the pipeline config."""
        return _SubprocessPipelineHandle(
            self._in_qs, self._out_q, self._max_workers, self._continuous
        )

    def _broadcast_shutdown(self) -> None:
        # Continuous: one shutdown per worker queue. Non-continuous: N onto the shared queue,
        # where each worker consumes exactly one (it stops pulling the instant it gets one).
        targets = (
            self._in_qs if self._continuous else [self._in_qs[0]] * self._max_workers
        )
        for q in targets:
            try:
                # Non-blocking: the queues are bounded and on the error path workers may have
                # stalled on a full ``_out_q`` (the bridge stage stops draining after relaying
                # an ``_ERROR``) and stopped consuming, so a blocking ``put`` would wedge here
                # forever and never reach the terminate/kill escalation below. A dropped
                # sentinel is fine — the worker that misses it will be terminated.
                q.put_nowait((_POOL_SHUTDOWN, None))
            except _queue.Full:
                # Expected miss when the queue is saturated; the missing worker is terminated
                # by the escalation below.
                continue
            except Exception:
                # Any other failure for one worker should not block the others; fall through
                # to the join/terminate/kill escalation.
                continue

    def shutdown(self) -> None:
        """Tell every worker to exit, then reap the processes (join → terminate → kill)."""
        self._broadcast_shutdown()
        self._terminate(self._procs)
        # Close this process's queue handles so the feeder threads created on first ``put``
        # exit; otherwise a process that creates many pipelines leaks threads and pipe fds.
        for q in (*self._in_qs, self._out_q):
            q.close()
            q.join_thread()


def _shutdown_pipeline_pools(pools: Sequence[_SubprocessPipelinePool]) -> None:
    for pool in pools:
        pool.shutdown()
