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
shared queues carried by a handle.

The wire protocol uses small integer message kinds (not sentinel objects): a sentinel pickled
onto a multiprocessing queue is a different object on the other side, so identity comparison
would not survive the trip.
"""

from __future__ import annotations

import asyncio
import queue as _queue
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any

from spdl.pipeline._common._misc import create_task

from ._common import is_eof
from ._queue import _queue_stage_hook, AsyncQueue

__all__ = [
    "_subprocess_pipeline",
    "_ITEM",
    "_SESSION_END",
    "_POOL_SHUTDOWN",
    "_RESULT",
    "_ERROR",
    "_DONE",
]

# Input-queue message kinds (this stage -> worker).
_ITEM = 0  # (_ITEM, value): one item for the sub-pipeline source
_SESSION_END = 1  # (_SESSION_END, None): end of one input stream; finish the session
_POOL_SHUTDOWN = 2  # (_POOL_SHUTDOWN, None): tear the worker down entirely

# Output-queue message kinds (worker -> this stage).
_RESULT = 0  # (_RESULT, value): one produced item
_ERROR = 1  # (_ERROR, exc): the sub-pipeline failed
_DONE = 2  # (_DONE, None): a worker finished the current session

# How long a blocking get may wait before yielding control, so the awaiting coroutine can
# observe cancellation between polls instead of parking a thread on an indefinite get.
_GET_TIMEOUT: float = 0.5


def _put(q: Any, msg: tuple[int, Any]) -> None:
    q.put(msg)


def _drain_one(q: Any) -> tuple[int, Any] | None:
    try:
        return q.get(timeout=_GET_TIMEOUT)
    except _queue.Empty:
        return None


async def _feed(
    input_queue: AsyncQueue,
    in_q: Any,
    num_workers: int,
    executor: Executor,
    abort: asyncio.Event,
) -> None:
    """Forward items from the stage's input queue to the worker pool, then end the session.

    Sends exactly one ``_SESSION_END`` per worker: a worker consumes exactly one such marker to
    end its current session, so the per-worker count ends every worker exactly once even though
    the input queue is shared. ``abort`` is set by :py:func:`_collect` on a worker error; once
    set, forwarding stops early and only the per-worker ``_SESSION_END`` markers are sent, so
    the workers wind down their current sessions instead of churning through the rest of the
    stream.

    The next-item ``get`` is raced against ``abort`` rather than awaited directly: on the error
    path the feeder is often parked here waiting on a slow/idle upstream, and ``abort`` must still
    interrupt it so the ``_SESSION_END`` markers go out. Otherwise the workers would block on
    ``in_q`` waiting for a marker that never arrives and :py:func:`_collect` would never see every
    ``_DONE`` — a hang bounded only by the collector's stall timeout. The pending item is dropped
    because the pipeline is already failing.
    """
    loop = asyncio.get_running_loop()
    abort_wait = create_task(abort.wait())
    try:
        while not abort.is_set():
            get_task = create_task(input_queue.get())
            try:
                await asyncio.wait(
                    {get_task, abort_wait}, return_when=asyncio.FIRST_COMPLETED
                )
                if not get_task.done():
                    break  # abort fired while parked on get; stop feeding
                item = get_task.result()
            finally:
                get_task.cancel()
            if is_eof(item):
                break
            await loop.run_in_executor(executor, _put, in_q, (_ITEM, item))
    finally:
        abort_wait.cancel()
    for _ in range(num_workers):
        await loop.run_in_executor(executor, _put, in_q, (_SESSION_END, None))


async def _collect(
    out_q: Any,
    num_workers: int,
    output_queue: AsyncQueue,
    executor: Executor,
    abort: asyncio.Event,
) -> None:
    """Forward worker results to the stage's output queue until every worker is done.

    On a worker ``_ERROR``, the first error is kept and ``abort`` is set (so the feeder stops
    sending new items), but draining continues until every worker has reported ``_DONE`` before
    the error is re-raised. Raising immediately would leave the still-running workers blocked on
    the bounded result queue with stale messages behind them, leaving the pool unusable without
    a full teardown. Results that arrive after the first error are discarded — the enclosing
    pipeline is already failing, and forwarding them could block on a no-longer-drained output
    queue.
    """
    loop = asyncio.get_running_loop()
    done = 0
    error: BaseException | None = None
    while done < num_workers:
        res = await loop.run_in_executor(executor, _drain_one, out_q)
        if res is None:
            continue  # timeout — loop so cancellation can be observed
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
    in_q, out_q, num_workers = handle.in_q, handle.out_q, handle.max_workers
    # Two threads: one parked on the feeder put, one on the collector get.
    executor = ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="spdl_fused_bridge_"
    )
    # Set by the collector on a worker error so the feeder stops forwarding new items promptly.
    abort = asyncio.Event()
    async with _queue_stage_hook(output_queue):
        feeder = create_task(_feed(input_queue, in_q, num_workers, executor, abort))
        collector = create_task(
            _collect(out_q, num_workers, output_queue, executor, abort)
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
