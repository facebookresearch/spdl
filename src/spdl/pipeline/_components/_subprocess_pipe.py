# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""The main-side bridge stage for a fused subprocess sub-pipeline.

A fused run of pipe stages (see :py:mod:`spdl.pipeline._fuse`) is replaced by a single stage
whose coroutine, defined here, streams items to a worker pool that runs the run as a nested
:py:class:`~spdl.pipeline.Pipeline`, and streams the results back. The pool itself lives in
:py:mod:`spdl.pipeline._subprocess_pipeline_pool`; this stage only talks to it through the
queues carried by a handle.

There are two modes, selected by ``handle.continuous``. Each worker has its own input queue in
both:

- **Non-continuous:** items are round-robined across the per-worker queues and one
  ``_SESSION_END`` is sent to each, so every worker ends its session exactly once; the stage
  finishes when every worker reports ``_DONE``. A per-worker queue (rather than one shared
  queue the workers steal from) is what guarantees a fast worker cannot consume a second
  ``_SESSION_END`` meant for a slower peer and let the stage finish before that peer flushes
  its items.
- **Continuous:** the source emits epoch boundaries (``_EPOCH_END``). Each worker's own input
  queue lets a boundary be broadcast to all of them, and the collector applies the same
  cross-stream barrier as :py:func:`spdl.pipeline._components._pipe._default_merge` — it emits one
  ``_EPOCH_END`` downstream only once every worker has reached the boundary. The feeder gates the
  next epoch's items behind that barrier so epochs stay correctly ordered across the pool.

The wire protocol uses small integer message kinds (not sentinel objects): a sentinel pickled
onto a multiprocessing queue is a different object on the other side, so identity comparison
would not survive the trip. ``_EPOCH_END`` itself never crosses — it is translated to the
``_EPOCH`` tag on the way in and re-emitted locally on the way out.
"""

from __future__ import annotations

import asyncio
import queue as _queue
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any

from spdl.pipeline._common._misc import create_task

from ._common import _EPOCH_END, is_eof, is_epoch_end
from ._queue import _queue_stage_hook, AsyncQueue

__all__ = [
    "_subprocess_pipeline",
    "_ITEM",
    "_SESSION_END",
    "_POOL_SHUTDOWN",
    "_EPOCH",
    "_RESULT",
    "_ERROR",
    "_DONE",
    "_EPOCH_DONE",
]

# Input-queue message kinds (this stage -> worker).
_ITEM = 0  # (_ITEM, value): one item for the sub-pipeline source
_SESSION_END = 1  # (_SESSION_END, None): end of one input stream; finish the session
_POOL_SHUTDOWN = 2  # (_POOL_SHUTDOWN, None): tear the worker down entirely
_EPOCH = 3  # (_EPOCH, None): end of the current epoch (continuous mode); keep the worker alive

# Output-queue message kinds (worker -> this stage).
_RESULT = 0  # (_RESULT, value): one produced item
_ERROR = 1  # (_ERROR, exc): the sub-pipeline failed
_DONE = 2  # (_DONE, None): a worker finished (session end, or exiting)
_EPOCH_DONE = (
    3  # (_EPOCH_DONE, None): a worker reached an epoch boundary (continuous mode)
)

# How long a blocking get may wait before yielding control, so the awaiting coroutine can
# observe cancellation between polls instead of parking a thread on an indefinite get.
_GET_TIMEOUT: float = 0.5

# Fixed bound (15 min) on how long the collector waits for any worker message before assuming a
# worker died abruptly and raising, instead of hanging forever. Comfortably above any per-stage
# latency in a data loader. Not user-configurable for now.
_WORKER_STALL_TIMEOUT: float = 900.0


def _put(q: Any, msg: tuple[int, Any]) -> None:
    q.put(msg)


def _drain_one(q: Any) -> tuple[int, Any] | None:
    try:
        return q.get(timeout=_GET_TIMEOUT)
    except _queue.Empty:
        return None


def _check_stall(last_progress: float) -> None:
    """Raise if no worker message has arrived within ``_WORKER_STALL_TIMEOUT`` seconds.

    A worker that dies abruptly (segfault, OOM-kill, external ``SIGKILL``) bypasses the worker
    loop's ``except`` and emits neither ``_ERROR`` nor ``_DONE``, so the collector would otherwise
    spin on empty reads forever and hang the enclosing pipeline. This bounds that wait. Any worker
    message (result, epoch boundary, done, error) counts as progress and resets the clock, so the
    bound only needs to exceed the slowest expected gap between messages.
    """
    if time.monotonic() - last_progress > _WORKER_STALL_TIMEOUT:
        raise TimeoutError(
            f"Fused subprocess stage received no worker message for "
            f"{_WORKER_STALL_TIMEOUT:.0f}s; a worker process may have died abruptly "
            "(e.g. segfault or OOM-kill)."
        )


########################################################################################
# Non-continuous: per-worker input queues, finish when all workers report _DONE.
########################################################################################


async def _feed(
    input_queue: AsyncQueue,
    in_qs: list[Any],
    executor: Executor,
    abort: asyncio.Event,
    feeder_idle: asyncio.Event,
) -> None:
    """Round-robin items across the per-worker queues, then end every worker's session.

    Sends exactly one ``_SESSION_END`` onto each worker's own queue, so every worker ends
    its session exactly once. Per-worker queues (rather than one shared queue the workers
    steal from) are what make this guarantee hold: on a shared queue a worker that reaches
    ``_SESSION_END`` early can loop back and consume a second marker meant for a slower peer
    still holding un-flushed items — the peer then never ends, :py:func:`_collect` reaches
    its ``_DONE`` count from the wrong workers and finishes, and the peer's items are
    silently dropped.

    ``abort`` is set by :py:func:`_collect` on a worker error; once set, forwarding stops early
    and only the per-worker ``_SESSION_END`` markers are sent, so the workers wind down their
    current sessions instead of churning through the rest of the stream.

    The next-item ``get`` is raced against ``abort`` rather than awaited directly: on the error
    path the feeder is often parked here waiting on a slow/idle upstream, and ``abort`` must still
    interrupt it so the ``_SESSION_END`` markers go out. Otherwise the workers would block on
    their queues waiting for a marker that never arrives and :py:func:`_collect` would never see
    every ``_DONE`` — a hang bounded only by the collector's stall timeout. The pending item is
    dropped because the pipeline is already failing.
    """
    loop = asyncio.get_running_loop()
    abort_wait = create_task(abort.wait())
    i = 0
    n = len(in_qs)
    try:
        while not abort.is_set():
            get_task = create_task(input_queue.get())
            feeder_idle.set()
            try:
                await asyncio.wait(
                    {get_task, abort_wait}, return_when=asyncio.FIRST_COMPLETED
                )
                if not get_task.done():
                    break  # abort fired while parked on get; stop feeding
                item = get_task.result()
            finally:
                get_task.cancel()
                feeder_idle.clear()
            if is_eof(item):
                break
            await loop.run_in_executor(executor, _put, in_qs[i % n], (_ITEM, item))
            i += 1
    finally:
        abort_wait.cancel()
    # Concurrent so a full/slow worker queue does not block the markers to the others.
    await asyncio.gather(
        *(loop.run_in_executor(executor, _put, q, (_SESSION_END, None)) for q in in_qs)
    )


async def _collect(
    out_q: Any,
    num_workers: int,
    output_queue: AsyncQueue,
    executor: Executor,
    abort: asyncio.Event,
    feeder_idle: asyncio.Event,
) -> None:
    """Forward worker results to the stage's output queue until every worker is done.

    On a worker ``_ERROR``, the first error is kept and ``abort`` is set (so the feeder stops
    sending new items), but draining continues until every worker has reported ``_DONE`` before
    the error is re-raised. Raising immediately would leave the still-running workers blocked on
    the bounded result queue with stale messages behind them, leaving the pool unusable without
    a full teardown. Results that arrive after the first error are discarded — the enclosing
    pipeline is already failing, and forwarding them could block on a no-longer-drained output
    queue.

    The stall guard is suppressed while ``feeder_idle`` is set: an idle feeder means nothing is
    dispatched and no worker message is due, so a quiet ``out_q`` is input starvation, not a
    dead worker.
    """
    loop = asyncio.get_running_loop()
    done = 0
    error: BaseException | None = None
    last_progress = time.monotonic()
    while done < num_workers:
        res = await loop.run_in_executor(executor, _drain_one, out_q)
        if res is None:
            if feeder_idle.is_set():
                last_progress = time.monotonic()  # input-starved, not a stalled worker
            else:
                _check_stall(last_progress)
            continue  # timeout — loop so cancellation can be observed
        last_progress = time.monotonic()
        kind, payload = res
        if kind == _DONE:
            done += 1
        elif kind == _ERROR:
            if error is None:
                error = payload
                abort.set()
        elif kind == _RESULT and error is None:
            await output_queue.put(payload)
    if error is not None:
        raise error


########################################################################################
# Continuous: per-worker input queues, epoch broadcast + barrier across the pool.
########################################################################################


async def _feed_continuous(
    input_queue: AsyncQueue,
    in_qs: list[Any],
    executor: Executor,
    epoch_barrier: asyncio.Event,
    feeder_idle: asyncio.Event,
) -> None:
    """Round-robin items to per-worker queues; broadcast and barrier each epoch boundary.

    On an epoch boundary the next epoch's items must not be fed until every worker has drained
    the current epoch (otherwise results from two epochs would interleave). The feeder therefore
    broadcasts ``_EPOCH`` to all workers and waits on ``epoch_barrier``, which the collector sets
    once it has emitted the epoch's single ``_EPOCH_END`` downstream.

    A single shared ``epoch_barrier`` is sufficient (rather than one per epoch) only because the
    feeder cannot advance past a boundary until the collector releases it: the feeder and
    collector strictly alternate one epoch at a time, so the ``clear()``/``set()`` never race.
    """
    loop = asyncio.get_running_loop()

    async def _broadcast(msg: tuple[int, Any]) -> None:
        # Concurrent so a full/slow worker queue does not block the broadcast to the others.
        await asyncio.gather(
            *(loop.run_in_executor(executor, _put, q, msg) for q in in_qs)
        )

    i = 0
    n = len(in_qs)
    while True:
        feeder_idle.set()
        item = await input_queue.get()
        feeder_idle.clear()
        if is_eof(item):
            await _broadcast((_POOL_SHUTDOWN, None))
            break
        if is_epoch_end(item):
            epoch_barrier.clear()
            await _broadcast((_EPOCH, None))
            await epoch_barrier.wait()
            continue
        await loop.run_in_executor(executor, _put, in_qs[i % n], (_ITEM, item))
        i += 1


async def _collect_continuous(
    out_q: Any,
    num_workers: int,
    output_queue: AsyncQueue,
    executor: Executor,
    epoch_barrier: asyncio.Event,
    feeder_idle: asyncio.Event,
) -> None:
    """Forward results; emit one ``_EPOCH_END`` per epoch once all workers reach the boundary.

    Mirrors the fan-in barrier in
    :py:func:`spdl.pipeline._components._pipe._default_merge`: count ``_EPOCH_DONE`` across the
    workers and, when all ``num_workers`` have reported, emit a single ``_EPOCH_END`` and release
    the feeder. Finishes when every worker reports ``_DONE`` (graceful shutdown); on normal
    pipeline stop this coroutine is cancelled instead.

    The stall guard is suppressed while ``feeder_idle`` is set (e.g. between epochs, waiting on a
    slow upstream for the next epoch's first item), where a quiet ``out_q`` is expected rather
    than a sign of a dead worker.
    """
    loop = asyncio.get_running_loop()
    workers_at_boundary = 0
    done = 0
    last_progress = time.monotonic()
    while done < num_workers:
        res = await loop.run_in_executor(executor, _drain_one, out_q)
        if res is None:
            if feeder_idle.is_set():
                last_progress = time.monotonic()  # input-starved, not a stalled worker
            else:
                _check_stall(last_progress)
            continue
        last_progress = time.monotonic()
        kind, payload = res
        if kind == _RESULT:
            await output_queue.put(payload)
        elif kind == _EPOCH_DONE:
            workers_at_boundary += 1
            if workers_at_boundary >= num_workers:
                workers_at_boundary = 0
                await output_queue.put(_EPOCH_END)
                epoch_barrier.set()
        elif kind == _ERROR:
            # Raise immediately rather than draining to every ``_DONE`` like :py:func:`_collect`.
            # A continuous worker only emits ``_ERROR`` on a fatal sub-pipeline failure and then
            # exits, so there is no warm pool left to preserve — the failure tears the whole
            # pipeline (and pool) down. Draining-to-``_DONE`` would also deadlock here: the other
            # workers stay warm waiting for the next epoch and never send ``_DONE`` without a
            # shutdown. Surviving workers are unblocked by the pool teardown that follows.
            raise payload
        elif kind == _DONE:
            done += 1


async def _subprocess_pipeline(
    input_queue: AsyncQueue, output_queue: AsyncQueue, handle: Any
) -> None:
    """Stream this stage's input through a worker pool and its results to the output queue.

    The fused run executes as a nested pipeline inside the pool's workers, so the handoff
    between the fused stages stays in-process. Worker failures are relayed as ``_ERROR`` and
    re-raised here so the enclosing pipeline fails as usual; the output queue's EOF is emitted
    by :py:func:`_queue_stage_hook` on both success and failure.

    The blocking multiprocessing-queue gets/puts run on a small dedicated thread pool so they do
    not occupy the enclosing pipeline's shared worker threads (which would starve its other
    stages).
    """
    in_qs, out_q = handle.in_qs, handle.out_q
    num_workers = handle.max_workers
    # One thread parked per concurrent blocking op: a put per input queue (the feeder broadcasts
    # epoch/session-end markers across all of them at once) plus the collector get.
    max_threads = len(in_qs) + 1
    executor = ThreadPoolExecutor(
        max_workers=max_threads, thread_name_prefix="spdl_fused_bridge_"
    )
    # Set by the feeder whenever it is parked waiting on the upstream queue. The collector reads
    # it to tell input starvation (no work dispatched, no worker message expected) apart from an
    # unresponsive worker, so its stall guard does not fire spuriously on a slow/idle source.
    feeder_idle = asyncio.Event()
    async with _queue_stage_hook(output_queue):
        if handle.continuous:
            epoch_barrier = asyncio.Event()
            feeder = create_task(
                _feed_continuous(
                    input_queue, in_qs, executor, epoch_barrier, feeder_idle
                )
            )
            collector = create_task(
                _collect_continuous(
                    out_q,
                    num_workers,
                    output_queue,
                    executor,
                    epoch_barrier,
                    feeder_idle,
                )
            )
        else:
            # Set by the collector on a worker error so the feeder stops forwarding new items.
            abort = asyncio.Event()
            feeder = create_task(
                _feed(input_queue, in_qs, executor, abort, feeder_idle)
            )
            collector = create_task(
                _collect(out_q, num_workers, output_queue, executor, abort, feeder_idle)
            )
        try:
            await asyncio.gather(feeder, collector)
        except BaseException:
            feeder.cancel()
            collector.cancel()
            await asyncio.gather(feeder, collector, return_exceptions=True)
            raise
        finally:
            # Don't wait on threads still parked in a blocking get/put — the pool teardown
            # unblocks them. ``cancel_futures`` discards anything not yet started.
            executor.shutdown(wait=False, cancel_futures=True)
