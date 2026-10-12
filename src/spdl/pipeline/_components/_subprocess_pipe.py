# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


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
- **Continuous:** the source emits epoch boundaries (``_EPOCH_END``). Each
  worker's own input queue lets a boundary be broadcast to all of them, and the
  collector applies the same cross-stream barrier as
  :py:func:`spdl.pipeline._components._pipe._default_merge` — it emits one
  ``_EPOCH_END`` downstream only once every worker has reached the boundary.
  The feeder gates the next epoch's items behind that barrier so epochs stay
  correctly ordered across the pool.

The wire protocol uses small integer message kinds (not sentinel objects): a sentinel pickled
onto a multiprocessing queue is a different object on the other side, so identity comparison
would not survive the trip. ``_EPOCH_END`` itself never crosses — it is translated to the
``_EPOCH`` tag on the way in and re-emitted locally on the way out.

``_ITEM`` and ``_RESULT`` payloads are **lists** of items, not single items, so a region whose
marker sets ``buffer_size=N`` amortizes the fixed per-transfer cost (the pickle and unpickle
calls, the queue feeder-thread handoff, the pipe write) over up to N items. Per-item costs such
as a tensor's shared-memory segment are unaffected. An unbuffered region simply sends
one-element lists. The packing and unpacking live entirely at this boundary — the feeder here
and the worker source/sink in :py:mod:`spdl.pipeline._subprocess_pipeline_pool` — so the
region's stages, and every stage outside it, still see individual items. Nothing assumes the two
directions use the same chunk size, or that the region's output cardinality matches its input's.
"""

from __future__ import annotations

import asyncio
import queue as _queue
import threading
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any, NoReturn, TypeGuard

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
    "_fused_queue_capacity",
]

# Input-queue message kinds (this stage -> worker).
_ITEM = 0  # (_ITEM, [value, ...]): one transfer of items for the sub-pipeline source
_SESSION_END = 1  # (_SESSION_END, None): end of one input stream; finish the session
_POOL_SHUTDOWN = 2  # (_POOL_SHUTDOWN, None): tear the worker down entirely
_EPOCH = 3  # (_EPOCH, epoch): end the current continuous epoch; keep the worker alive

# Output-queue message kinds (worker -> this stage).
_RESULT = 0  # (_RESULT, [value, ...]): one transfer of produced items
_ERROR = 1  # (_ERROR, exc): the sub-pipeline failed
_DONE = 2  # (_DONE, None): a worker finished (session end, or exiting)
_EPOCH_DONE = 3  # (_EPOCH_DONE, (epoch, worker)): worker reached a continuous boundary

_SESSION_OUTPUT_KINDS: tuple[int, ...] = (_RESULT, _ERROR, _DONE)
_CONTINUOUS_OUTPUT_KINDS: tuple[int, ...] = (*_SESSION_OUTPUT_KINDS, _EPOCH_DONE)

# How long a blocking get may wait before yielding control, so the awaiting coroutine can
# observe cancellation between polls instead of parking a thread on an indefinite get.
_GET_TIMEOUT: float = 0.5

# How long a blocking put may wait before re-checking the teardown flag. A put onto a *full*
# worker queue (backpressure once the consumer stops) cannot be released by pool teardown -- an
# mp.Queue putter blocked on the queue's semaphore is not woken by closing the queue or
# terminating the consumer workers -- so an indefinite put would park this pool
# thread forever and hang interpreter exit (a non-daemon executor thread joined
# by concurrent.futures at shutdown).
_PUT_TIMEOUT: float = 0.5

# Fixed bound (15 min) on how long the collector waits for any worker message before assuming a
# worker died abruptly and raising, instead of hanging forever. Comfortably above any per-stage
# latency in a data loader. Not user-configurable for now.
_WORKER_STALL_TIMEOUT: float = 900.0

# Give a multiprocessing queue's feeder time to flush the detailed fallback error
# before synthesizing a generic one. The wait remains bounded so a failed fallback
# cannot defer the error until the much longer worker-stall timeout.
_SERIALIZATION_ERROR_RELAY_TIMEOUT: float = 5.0
_BUFFERED_ERROR_SCAN_TIMEOUT: float = 0.5
_PROTOCOL_DIAGNOSTIC_LIMIT: int = 128

_SERIALIZATION_ERROR_RELAY_FAILURE = (
    "Fused subprocess {boundaries} serialization failed, but the detailed error "
    "could not be relayed through the worker output queue."
)
_QUEUE_FEEDER_FAILURE = (
    "Fused subprocess {boundaries} queue feeder failed while sending a protocol "
    "or transport message."
)


class _WorkerOutputQueueError(RuntimeError):
    """The worker output queue could no longer be read."""


def _fused_queue_capacity(num_workers: int) -> int:
    """Return the shared message capacity for each fused worker queue."""
    return max(4, num_workers * 2)


def _put(q: Any, msg: tuple[int, Any], stop: threading.Event) -> None:
    # Bounded, interruptible put: poll so this pool thread wakes to observe teardown (``stop``)
    # and exit, rather than parking forever on a full queue (see ``_PUT_TIMEOUT``).
    while not stop.is_set():
        try:
            q.put(msg, timeout=_PUT_TIMEOUT)
            return
        except _queue.Full:
            continue
    # A synchronous-serialization fallback retains the original object so a
    # bounded retry does not rerun its reducer. Teardown is the one path that
    # abandons rather than retries that object, so release the retained state.
    discard_pending = getattr(q, "_discard_pending", None)
    if callable(discard_pending):
        discard_pending(msg)


def _drain_one(q: Any) -> tuple[int, Any] | None:
    try:
        return q.get(timeout=_GET_TIMEOUT)
    except _queue.Empty:
        return None
    except (EOFError, OSError, RuntimeError, ValueError) as error:
        raise _WorkerOutputQueueError(
            "The fused subprocess worker output queue could not be read."
        ) from error


def _is_valid_worker_message(
    message: Any, allowed_kinds: tuple[int, ...]
) -> TypeGuard[tuple[int, Any]]:
    """Validate the fused worker output protocol before dispatching a message."""
    if type(message) is not tuple or len(message) != 2:
        return False
    kind, payload = message
    if type(kind) is not int or kind not in allowed_kinds:
        return False
    if kind == _RESULT:
        return type(payload) is list
    if kind == _ERROR:
        return issubclass(type(payload), BaseException)
    if kind == _EPOCH_DONE:
        return (
            type(payload) is tuple
            and len(payload) == 2
            and type(payload[0]) is int
            and type(payload[1]) is int
        )
    return payload is None


def _bounded_protocol_type_name(value: Any) -> str:
    """Return a bounded type name without trusting user-defined metaclasses."""
    try:
        name = type(value).__name__
        if type(name) is not str:
            return "<type unavailable>"
    except BaseException:  # noqa: B036 - diagnostics must survive hostile types
        return "<type unavailable>"
    if len(name) > _PROTOCOL_DIAGNOSTIC_LIMIT:
        return name[:_PROTOCOL_DIAGNOSTIC_LIMIT] + "... <truncated>"
    return name


def _malformed_worker_message_reason(
    message: Any, allowed_kinds: tuple[int, ...]
) -> str:
    """Describe malformed protocol structure without rendering its payload."""
    if type(message) is not tuple:
        return f"expected a 2-tuple, got {_bounded_protocol_type_name(message)}"
    if len(message) != 2:
        return f"expected 2 fields, got {len(message)}"
    kind, payload = message
    if type(kind) is not int:
        return (
            f"message kind must be an integer, got {_bounded_protocol_type_name(kind)}"
        )
    if kind not in allowed_kinds:
        if kind.bit_length() > 64:
            return "unexpected integer message kind outside 64-bit range"
        return f"unexpected message kind {kind}"
    if kind == _RESULT:
        return (
            f"RESULT payload must be a list, got {_bounded_protocol_type_name(payload)}"
        )
    if kind == _ERROR:
        return (
            "ERROR payload must be an exception, got "
            f"{_bounded_protocol_type_name(payload)}"
        )
    if kind == _EPOCH_DONE:
        if type(payload) is not tuple:
            return (
                "EPOCH_DONE payload must be an (epoch, worker) tuple, got "
                f"{_bounded_protocol_type_name(payload)}"
            )
        if len(payload) != 2:
            return f"EPOCH_DONE payload must have 2 fields, got {len(payload)}"
        epoch, worker = payload
        if type(epoch) is not int or type(worker) is not int:
            return "EPOCH_DONE epoch and worker fields must be integers"
        return "EPOCH_DONE payload is invalid"
    return (
        "control-message payload must be None, got "
        f"{_bounded_protocol_type_name(payload)}"
    )


def _check_stall(
    last_progress: float, prior_error: BaseException | None = None
) -> None:
    """Raise a prior worker error or timeout after a prolonged message gap.

    A worker that dies abruptly (segfault, OOM-kill, external ``SIGKILL``)
    bypasses the worker loop's ``except`` and emits neither ``_ERROR`` nor
    ``_DONE``, so the collector would otherwise spin on empty reads forever and
    hang the enclosing pipeline. This bounds that wait. Any worker message
    (result, epoch boundary, done, error) counts as progress and resets the
    clock, so the bound only needs to exceed the slowest expected gap between
    messages.
    """
    if time.monotonic() - last_progress > _WORKER_STALL_TIMEOUT:
        stall_error = TimeoutError(
            f"Fused subprocess stage received no worker message for "
            f"{_WORKER_STALL_TIMEOUT:.0f}s; a worker process may have died abruptly "
            "(e.g. segfault or OOM-kill)."
        )
        if prior_error is not None:
            raise prior_error
        raise stall_error


def _serialization_failure_boundaries(
    serialization_failures: tuple[tuple[str, Any], ...],
) -> str | None:
    """Return a display string listing all failed queue boundaries, if any."""
    boundaries = [
        boundary
        for boundary, failure in serialization_failures
        if failure is not None and failure.failed.is_set()
    ]
    return " and ".join(boundaries) or None


def _serialization_relay_failure_boundaries(
    serialization_failures: tuple[tuple[str, Any], ...],
) -> str | None:
    """Return boundaries whose detailed fallback could not be queued."""
    boundaries = [
        boundary
        for boundary, failure in serialization_failures
        if failure is not None and failure.relay_failed.is_set()
    ]
    return " and ".join(boundaries) or None


def _non_payload_failure_boundaries(
    serialization_failures: tuple[tuple[str, Any], ...],
) -> str | None:
    """Return boundaries whose feeder failed outside a payload pickle."""
    boundaries = [
        boundary
        for boundary, failure in serialization_failures
        if failure is not None and failure.non_payload_failed.is_set()
    ]
    return " and ".join(boundaries) or None


def _non_payload_feeder_error(
    serialization_failures: tuple[tuple[str, Any], ...],
) -> RuntimeError | None:
    """Describe a control-message or transport failure reported by a feeder."""
    boundaries = _non_payload_failure_boundaries(serialization_failures)
    if boundaries is None:
        return None
    return RuntimeError(_QUEUE_FEEDER_FAILURE.format(boundaries=boundaries))


def _serialization_pending_relay_boundaries(
    serialization_failures: tuple[tuple[str, Any], ...],
) -> str | None:
    """Return failed boundaries whose detailed fallback may still arrive."""
    boundaries = [
        boundary
        for boundary, failure in serialization_failures
        if failure is not None
        and failure.failed.is_set()
        and not failure.relay_failed.is_set()
    ]
    return " and ".join(boundaries) or None


def _collection_is_complete(
    done: int,
    num_workers: int,
    serialization_failures: tuple[tuple[str, Any], ...],
    error: BaseException | None = None,
) -> bool:
    """Return whether all workers finished without feeder detail still due."""
    return done >= num_workers and (
        error is not None
        or _serialization_failure_boundaries(serialization_failures) is None
    )


def _record_epoch_boundary(
    payload: tuple[int, int],
    expected_epoch: int,
    num_workers: int,
    workers_at_boundary: set[int],
) -> bool:
    """Record one generation-tagged worker boundary and reject stale duplicates."""
    epoch, worker = payload
    if epoch != expected_epoch:
        raise RuntimeError("A worker reported a boundary for an unexpected epoch.")
    if worker < 0 or worker >= num_workers:
        raise RuntimeError("An unknown worker reported an epoch boundary.")
    if worker in workers_at_boundary:
        raise RuntimeError(
            "A worker reported more than one boundary for the current epoch."
        )
    workers_at_boundary.add(worker)
    return len(workers_at_boundary) == num_workers


def _serialization_error_after_queue_teardown(
    serialization_failures: tuple[tuple[str, Any], ...],
) -> RuntimeError | None:
    """Describe a known feeder failure whose detail queue became unreadable."""
    if (feeder_error := _non_payload_feeder_error(serialization_failures)) is not None:
        return feeder_error
    failure_boundaries = _serialization_failure_boundaries(serialization_failures)
    if failure_boundaries is None:
        return None
    return RuntimeError(
        _SERIALIZATION_ERROR_RELAY_FAILURE.format(boundaries=failure_boundaries)
    )


def _raise_worker_output_queue_error(
    queue_error: _WorkerOutputQueueError,
    serialization_failures: tuple[tuple[str, Any], ...],
    worker_error: BaseException | None = None,
) -> NoReturn:
    """Raise the most specific failure known when the worker queue closes."""
    if worker_error is not None:
        raise worker_error from None
    if (
        serialization_error := _serialization_error_after_queue_teardown(
            serialization_failures
        )
    ) is not None:
        raise serialization_error from queue_error
    raise queue_error


def _check_serialization_error_relay(
    failure_boundaries: str | None, deadline: float | None
) -> float | None:
    """Bound how long a collector waits for a detailed feeder error."""
    if failure_boundaries is None:
        return None
    now = time.monotonic()
    if deadline is None:
        return now + _SERIALIZATION_ERROR_RELAY_TIMEOUT
    if now >= deadline:
        raise RuntimeError(
            _SERIALIZATION_ERROR_RELAY_FAILURE.format(boundaries=failure_boundaries)
        )
    return deadline


def _pop_buffered_serialization_error(q: Any, timeout: float) -> BaseException | None:
    """Wait for buffered detail while bounding a concurrently replenished drain."""
    deadline = time.monotonic() + timeout
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            message = q.get(timeout=remaining)
        except (_queue.Empty, EOFError, OSError, RuntimeError, ValueError):
            return None
        if not _is_valid_worker_message(message, (_ERROR,)):
            continue
        _, payload = message
        return payload
    return None


async def _check_serialization_error_relay_after_progress(
    out_q: Any,
    executor: Executor,
    serialization_failures: tuple[tuple[str, Any], ...],
    deadline: float | None,
) -> float | None:
    """Prefer a buffered detailed error before raising the generic relay failure."""
    feeder_error = _non_payload_feeder_error(serialization_failures)
    generic_error: RuntimeError | None = None
    pending_boundaries = _serialization_pending_relay_boundaries(serialization_failures)
    if pending_boundaries is not None:
        failure_boundaries = _serialization_failure_boundaries(serialization_failures)
        try:
            deadline = _check_serialization_error_relay(failure_boundaries, deadline)
        except RuntimeError as error:
            generic_error = error
    elif feeder_error is not None:
        raise feeder_error
    elif (
        relay_failure_boundaries := _serialization_relay_failure_boundaries(
            serialization_failures
        )
    ) is not None:
        generic_error = RuntimeError(
            _SERIALIZATION_ERROR_RELAY_FAILURE.format(
                boundaries=relay_failure_boundaries
            )
        )
    else:
        return None
    if generic_error is None:
        return deadline

    # Producers can refill slots while this scan runs, so a message-count bound can
    # miss detail already queued behind their messages. Drain until the queue is empty,
    # with a time bound in case producers keep it continuously non-empty. The stage is
    # already failing, so other buffered results/markers can be discarded.
    loop = asyncio.get_running_loop()
    detailed_error = await loop.run_in_executor(
        executor,
        _pop_buffered_serialization_error,
        out_q,
        _BUFFERED_ERROR_SCAN_TIMEOUT,
    )
    if detailed_error is not None:
        raise detailed_error
    if feeder_error is not None:
        raise feeder_error
    raise generic_error


async def _wait_for_serialization_error_after_malformed_message(
    message: Any,
    allowed_kinds: tuple[int, ...],
    out_q: Any,
    executor: Executor,
    serialization_failures: tuple[tuple[str, Any], ...],
    deadline: float | None,
) -> float:
    """Wait for a flagged feeder error instead of exposing malformed queue noise."""
    deadline = await _check_serialization_error_relay_after_progress(
        out_q,
        executor,
        serialization_failures,
        deadline,
    )
    if deadline is None:
        reason = _malformed_worker_message_reason(message, allowed_kinds)
        raise RuntimeError(
            f"A fused subprocess worker sent a malformed message: {reason}."
        )
    return deadline


########################################################################################
# Non-continuous: per-worker input queues, finish when all workers report _DONE.
########################################################################################


async def _feed(
    input_queue: AsyncQueue,
    in_qs: list[Any],
    executor: Executor,
    abort: asyncio.Event,
    feeder_idle: asyncio.Event,
    put_stop: threading.Event,
    buffer_size: int = 1,
) -> None:
    """Round-robin transfers across the per-worker queues, then end every worker's session.

    Items are accumulated into a chunk of up to ``buffer_size`` and sent as one
    ``_ITEM`` message; whole chunks, not individual items, are what round-robin across the
    workers. The tail chunk is flushed before the end markers, so no item is left behind.

    Accumulation blocks: filling a chunk means waiting for ``buffer_size`` items.
    Flushing early on a momentarily empty upstream is pointless here, because the inter-stage
    queues are only two deep — a non-blocking drain could never gather more than two or three
    items and ``buffer_size`` would have no effect.

    Sends exactly one ``_SESSION_END`` onto each worker's own queue, so every worker ends
    its session exactly once. Per-worker queues (rather than one shared queue the workers
    steal from) are what make this guarantee hold: on a shared queue a worker that reaches
    ``_SESSION_END`` early can loop back and consume a second marker meant for a slower peer
    still holding un-flushed items — the peer then never ends, :py:func:`_collect` reaches
    its ``_DONE`` count from the wrong workers and finishes, and the peer's items are
    silently dropped.

    ``abort`` is set by :py:func:`_collect` on a worker error; once set, forwarding stops early
    and only the per-worker ``_SESSION_END`` markers are sent, so the workers wind down their
    current sessions instead of churning through the rest of the stream. A partly-filled chunk
    is dropped on that path along with everything else still in flight.

    The next-item ``get`` is raced against ``abort`` rather than awaited directly: on the error
    path the feeder is often parked here waiting on a slow/idle upstream, and
    ``abort`` must still interrupt it so the ``_SESSION_END`` markers go out.
    Otherwise the workers would block on
    their queues waiting for a marker that never arrives and :py:func:`_collect` would never see
    every ``_DONE`` — a hang bounded only by the collector's stall timeout. The pending item is
    dropped because the pipeline is already failing.
    """
    loop = asyncio.get_running_loop()
    abort_wait = create_task(abort.wait())
    i = 0
    n = len(in_qs)
    buf: list[Any] = []
    reached_eof = False

    async def _flush() -> None:
        nonlocal buf, i
        if not buf:
            return
        chunk, buf = buf, []
        await loop.run_in_executor(
            executor, _put, in_qs[i % n], (_ITEM, chunk), put_stop
        )
        i += 1

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
                reached_eof = True
                break
            buf.append(item)
            if len(buf) >= buffer_size:
                await _flush()
    finally:
        abort_wait.cancel()
    # Only the clean end-of-stream flushes its tail. Keyed on ``reached_eof`` rather than on
    # ``abort`` so an abort racing the EOF break cannot discard a legitimate tail chunk.
    if reached_eof:
        await _flush()
    # Concurrent so a full/slow worker queue does not block the markers to the others.
    await asyncio.gather(
        *(
            loop.run_in_executor(executor, _put, q, (_SESSION_END, None), put_stop)
            for q in in_qs
        )
    )


async def _collect(
    out_q: Any,
    num_workers: int,
    output_queue: AsyncQueue,
    executor: Executor,
    abort: asyncio.Event,
    feeder_idle: asyncio.Event,
    serialization_failures: tuple[tuple[str, Any], ...] = (),
) -> None:
    """Forward worker results to the stage's output queue until every worker is done.

    On a worker ``_ERROR``, the first error is kept and ``abort`` is set (so the feeder stops
    sending new items), but draining continues until every worker has reported ``_DONE`` before
    the error is re-raised. Raising immediately would leave the still-running workers blocked on
    the bounded result queue with stale messages behind them, leaving the pool unusable without
    a full teardown. The one exception is a failed serialization-error relay: its dropped
    protocol message may itself be a ``_DONE`` marker, so collection exits after the bounded
    relay check and lets pool teardown reap the workers. Results that arrive after the first
    error are discarded — the enclosing pipeline is already failing, and forwarding them could
    block on a no-longer-drained output queue.

    Each ``_RESULT`` carries a list of items, which is unpacked onto the output queue one item
    at a time so the region's buffering stays invisible downstream.

    The stall guard is suppressed while ``feeder_idle`` is set: an idle feeder means nothing is
    dispatched and no worker message is due, so a quiet ``out_q`` is input starvation, not a
    dead worker.
    """
    loop = asyncio.get_running_loop()
    done = 0
    error: BaseException | None = None
    last_progress = time.monotonic()
    serialization_error_deadline: float | None = None
    while True:
        try:
            res = await loop.run_in_executor(executor, _drain_one, out_q)
        except _WorkerOutputQueueError as queue_error:
            _raise_worker_output_queue_error(queue_error, serialization_failures, error)
        if res is None:
            if (
                error is not None
                and _serialization_relay_failure_boundaries(serialization_failures)
                is not None
            ):
                break
            serialization_error_deadline = (
                await _check_serialization_error_relay_after_progress(
                    out_q,
                    executor,
                    serialization_failures if error is None else (),
                    serialization_error_deadline,
                )
            )
            if serialization_error_deadline is not None:
                abort.set()
                continue  # the bounded relay grace supersedes the stall guard
            if feeder_idle.is_set() and error is None:
                last_progress = time.monotonic()  # input-starved, not stalled
            else:
                _check_stall(last_progress, error)
            continue  # timeout — loop so cancellation can be observed
        last_progress = time.monotonic()
        if not _is_valid_worker_message(res, _SESSION_OUTPUT_KINDS):
            if error is not None:
                continue
            abort.set()
            serialization_error_deadline = (
                await _wait_for_serialization_error_after_malformed_message(
                    res,
                    _SESSION_OUTPUT_KINDS,
                    out_q,
                    executor,
                    serialization_failures,
                    serialization_error_deadline,
                )
            )
            continue
        kind, payload = res
        failure_pending = False
        failures = serialization_failures if error is None else ()
        if kind != _ERROR and (
            serialization_error_deadline is not None
            or _serialization_failure_boundaries(failures) is not None
        ):
            serialization_error_deadline = (
                await _check_serialization_error_relay_after_progress(
                    out_q,
                    executor,
                    failures,
                    serialization_error_deadline,
                )
            )
            failure_pending = serialization_error_deadline is not None
            if failure_pending:
                abort.set()
        done_delta, error = await _handle_collected_result(
            kind,
            payload,
            output_queue,
            abort,
            error,
            failure_pending,
        )
        done += done_delta
        if _collection_is_complete(done, num_workers, serialization_failures, error):
            break
    if error is not None:
        raise error


async def _handle_collected_result(
    kind: int,
    payload: Any,
    output_queue: AsyncQueue,
    abort: asyncio.Event,
    error: BaseException | None,
    failure_pending: bool,
) -> tuple[int, BaseException | None]:
    """Handle one valid non-continuous worker message."""
    if kind == _DONE:
        return 1, error
    if kind == _ERROR:
        if error is None:
            abort.set()
            return 0, payload
        return 0, error
    if kind == _RESULT and error is None and not failure_pending:
        for item in payload:
            await output_queue.put(item)
    return 0, error


########################################################################################
# Continuous: per-worker input queues, epoch broadcast + barrier across the pool.
########################################################################################


async def _feed_continuous(
    input_queue: AsyncQueue,
    in_qs: list[Any],
    executor: Executor,
    epoch_barrier: asyncio.Event,
    feeder_idle: asyncio.Event,
    put_stop: threading.Event,
    buffer_size: int = 1,
) -> None:
    """Round-robin transfers to per-worker queues; broadcast and barrier each epoch boundary.

    Items are accumulated into chunks of up to ``buffer_size`` and sent as one
    ``_ITEM`` message each, exactly as in :py:func:`_feed`.

    On an epoch boundary the next epoch's items must not be fed until every
    worker has drained the current epoch (otherwise results from two epochs
    would interleave). The feeder therefore broadcasts ``_EPOCH`` to all
    workers and waits on ``epoch_barrier``, which the collector sets once it
    has emitted the epoch's single ``_EPOCH_END`` downstream.

    A partly-filled chunk **must** be flushed before that broadcast: an ``_ITEM`` put after the
    ``_EPOCH`` marker would be drained by the worker as part of the *next* epoch, so the tail of
    one epoch would silently reappear in the following one. Flushing first keeps the chunk ahead
    of the marker on each worker's FIFO queue. The same applies to the ``_POOL_SHUTDOWN``
    broadcast at end of stream.

    A single shared ``epoch_barrier`` is sufficient (rather than one per epoch) only because the
    feeder cannot advance past a boundary until the collector releases it: the feeder and
    collector strictly alternate one epoch at a time, so the ``clear()``/``set()`` never race.
    """
    loop = asyncio.get_running_loop()

    async def _broadcast(msg: tuple[int, Any]) -> None:
        # Concurrent so a full/slow worker queue does not block the broadcast to the others.
        await asyncio.gather(
            *(loop.run_in_executor(executor, _put, q, msg, put_stop) for q in in_qs)
        )

    i = 0
    n = len(in_qs)
    epoch = 0
    buf: list[Any] = []

    async def _flush() -> None:
        nonlocal buf, i
        if not buf:
            return
        chunk, buf = buf, []
        await loop.run_in_executor(
            executor, _put, in_qs[i % n], (_ITEM, chunk), put_stop
        )
        i += 1

    while True:
        feeder_idle.set()
        item = await input_queue.get()
        feeder_idle.clear()
        if is_eof(item):
            await _flush()
            await _broadcast((_POOL_SHUTDOWN, None))
            break
        if is_epoch_end(item):
            await _flush()
            epoch_barrier.clear()
            await _broadcast((_EPOCH, epoch))
            await epoch_barrier.wait()
            epoch += 1
            continue
        buf.append(item)
        if len(buf) >= buffer_size:
            await _flush()


async def _collect_continuous(
    out_q: Any,
    num_workers: int,
    output_queue: AsyncQueue,
    executor: Executor,
    epoch_barrier: asyncio.Event,
    feeder_idle: asyncio.Event,
    serialization_failures: tuple[tuple[str, Any], ...] = (),
) -> None:
    """Forward results; emit one ``_EPOCH_END`` per epoch once all workers reach the boundary.

    Mirrors the fan-in barrier in
    :py:func:`spdl.pipeline._components._pipe._default_merge`: count
    ``_EPOCH_DONE`` across the workers and, when all ``num_workers`` have
    reported, emit a single ``_EPOCH_END`` and release the feeder. Finishes
    when every worker reports ``_DONE`` (graceful shutdown); on normal pipeline
    stop this coroutine is cancelled instead.

    The stall guard is suppressed while ``feeder_idle`` is set (e.g. between
    epochs, waiting on a slow upstream for the next epoch's first item), where
    a quiet ``out_q`` is expected rather than a sign of a dead worker.
    """
    loop = asyncio.get_running_loop()
    workers_at_boundary: set[int] = set()
    expected_epoch = 0
    done = 0
    last_progress = time.monotonic()
    serialization_error_deadline: float | None = None
    while True:
        try:
            res = await loop.run_in_executor(executor, _drain_one, out_q)
        except _WorkerOutputQueueError as queue_error:
            _raise_worker_output_queue_error(queue_error, serialization_failures)
        if res is None:
            serialization_error_deadline = (
                await _check_serialization_error_relay_after_progress(
                    out_q,
                    executor,
                    serialization_failures,
                    serialization_error_deadline,
                )
            )
            if serialization_error_deadline is not None:
                continue  # the bounded relay grace supersedes the stall guard
            if feeder_idle.is_set():
                last_progress = time.monotonic()  # input-starved, not stalled
            else:
                _check_stall(last_progress)
            continue
        last_progress = time.monotonic()
        if not _is_valid_worker_message(res, _CONTINUOUS_OUTPUT_KINDS):
            serialization_error_deadline = (
                await _wait_for_serialization_error_after_malformed_message(
                    res,
                    _CONTINUOUS_OUTPUT_KINDS,
                    out_q,
                    executor,
                    serialization_failures,
                    serialization_error_deadline,
                )
            )
            continue
        kind, payload = res
        failure_pending = False
        if kind != _ERROR and (
            serialization_error_deadline is not None
            or _serialization_failure_boundaries(serialization_failures) is not None
        ):
            serialization_error_deadline = (
                await _check_serialization_error_relay_after_progress(
                    out_q,
                    executor,
                    serialization_failures,
                    serialization_error_deadline,
                )
            )
            failure_pending = serialization_error_deadline is not None
        if kind == _RESULT and not failure_pending:
            for item in payload:
                await output_queue.put(item)
        elif kind == _EPOCH_DONE:
            # A failed result can be followed by its already-buffered boundary marker before
            # the fallback _ERROR. Keep the feeder blocked and drain for that terminal detail.
            if failure_pending:
                continue
            if _record_epoch_boundary(
                payload,
                expected_epoch,
                num_workers,
                workers_at_boundary,
            ):
                workers_at_boundary.clear()
                expected_epoch += 1
                await output_queue.put(_EPOCH_END)
                epoch_barrier.set()
        elif kind == _ERROR:
            # Raise immediately rather than draining to every ``_DONE`` like
            # :py:func:`_collect`. A continuous worker only emits ``_ERROR`` on a
            # fatal sub-pipeline failure and then exits, so there is no warm pool
            # left to preserve — the failure tears the whole pipeline (and pool)
            # down. Draining-to-``_DONE`` would also deadlock here: the other
            # workers stay warm waiting for the next epoch and never send
            # ``_DONE`` without a shutdown. Surviving workers are unblocked by
            # the pool teardown that follows.
            raise payload
        elif kind == _DONE:
            done += 1
        if _collection_is_complete(done, num_workers, serialization_failures):
            break


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
    handle.prepare_bridge()
    in_qs, out_q = handle.in_qs, handle.out_q
    num_workers = handle.max_workers
    serialization_failures = (
        ("input", handle.input_serialization_failed),
        ("output", handle.output_serialization_failed),
    )
    # Only the inbound size matters here; the worker owns the outbound one.
    buffer_size = handle.input_buffer_size
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
    # Signals threads parked in a blocking ``_put`` to stop and exit on teardown.
    # A put onto a full worker queue cannot be released by pool teardown (see
    # ``_PUT_TIMEOUT``), so without this signal its thread would outlive this
    # stage and hang interpreter exit.
    put_stop = threading.Event()
    async with _queue_stage_hook(output_queue):
        if handle.continuous:
            epoch_barrier = asyncio.Event()
            feeder = create_task(
                _feed_continuous(
                    input_queue,
                    in_qs,
                    executor,
                    epoch_barrier,
                    feeder_idle,
                    put_stop,
                    buffer_size,
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
                    serialization_failures,
                )
            )
        else:
            # Set by the collector on a worker error so the feeder stops forwarding new items.
            abort = asyncio.Event()
            feeder = create_task(
                _feed(
                    input_queue,
                    in_qs,
                    executor,
                    abort,
                    feeder_idle,
                    put_stop,
                    buffer_size,
                )
            )
            collector = create_task(
                _collect(
                    out_q,
                    num_workers,
                    output_queue,
                    executor,
                    abort,
                    feeder_idle,
                    serialization_failures,
                )
            )
        try:
            await asyncio.gather(feeder, collector)
        except BaseException:
            feeder.cancel()
            collector.cancel()
            await asyncio.gather(feeder, collector, return_exceptions=True)
            raise
        finally:
            # Release any thread parked in a blocking ``_put`` so it exits instead of outliving
            # this stage, then drop the executor. ``cancel_futures`` discards anything not yet
            # started; a still-parked get self-releases within ``_GET_TIMEOUT``.
            put_stop.set()
            executor.shutdown(wait=False, cancel_futures=True)
