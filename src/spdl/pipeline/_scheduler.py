# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Priority-dispatch scheduler for SPDL sync stages.

When enabled via ``use_priority_scheduler=True`` on ``build_pipeline()``,
sync ops are routed through a priority queue instead of the underlying
``ThreadPoolExecutor``'s FIFO. Priority is ``-depth``: deeper stages
(closer to the sink) are dispatched first, draining the pipeline.

Architecture (v5)
-----------------

The hot path is a thin :py:class:`_PrioritizedExecutor` shim that
implements the :py:class:`concurrent.futures.Executor` ABC. Each sync
stage gets its own shim, captured as the stage's ``_PipeArgs.executor``.
The shim's :py:meth:`~_PrioritizedExecutor.submit` enqueues a
:py:class:`_WorkItem` onto a shared heap inside :py:class:`PriorityScheduler`,
which dispatches in priority order to a single underlying
:py:class:`~concurrent.futures.ThreadPoolExecutor`.

The scheduler runs as a :py:class:`~spdl.pipeline._bg_task.BackgroundTask`
on the pipeline's event loop via the
:py:class:`_PrioritySchedulerBackgroundTask` adapter.

Cancellation semantics
----------------------

``cf_future.set_running_or_notify_cancel()`` is called at *dispatch*
time inside :py:meth:`PriorityScheduler.run` (NOT at submit time). This
preserves the standard ``concurrent.futures.Future`` contract: pre-dispatch
``cancel()`` returns ``True`` and the dispatch loop skips the work item;
post-dispatch ``cancel()`` returns ``False`` and the work completes
normally (matching :py:class:`~concurrent.futures.ThreadPoolExecutor`).
"""

from __future__ import annotations

__all__ = [
    "PriorityScheduler",
    "_PrioritizedExecutor",
    "_PrioritySchedulerBackgroundTask",
]

import asyncio
import concurrent.futures
import heapq
import logging
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, cast

from spdl.pipeline._bg_task import BackgroundTask
from spdl.pipeline._common._types import StageInfo

_LG: logging.Logger = logging.getLogger(__name__)


@dataclass(order=True)
class _WorkItem:
    """Unit of work ordered by ``(priority, seq)`` for heap dispatch.

    ``cf_future`` is the :py:class:`~concurrent.futures.Future` returned
    from :py:meth:`_PrioritizedExecutor.submit`. ``bridge`` is a small
    mutable dict that holds the dispatched :py:class:`asyncio.Future`
    and a ``dispatched`` flag so that post-dispatch cancellation can
    forward to the underlying task.
    """

    priority: int
    seq: int
    func: Callable[..., Any] = field(compare=False)
    args: tuple[Any, ...] = field(compare=False)
    kwargs: dict[str, Any] = field(compare=False)
    cf_future: concurrent.futures.Future[Any] = field(compare=False)
    bridge: dict[str, Any] = field(compare=False)


class PriorityScheduler:
    """Priority-based dispatch scheduler for sync pipeline stages.

    The scheduler keeps a min-heap of :py:class:`_WorkItem`s. The
    :py:meth:`run` coroutine pops items in ``(priority, seq)`` order and
    dispatches each to the underlying :py:class:`ThreadPoolExecutor`,
    bridging the result back to the caller's ``cf_future`` via
    :py:meth:`asyncio.AbstractEventLoop.call_soon_threadsafe`.

    All scheduler bookkeeping mutations happen on the event-loop thread.

    Args:
        max_concurrent: Maximum simultaneous dispatches to the underlying
            pool (typically ``== num_threads``).

    Note:
        The underlying :py:class:`ThreadPoolExecutor` is bound by
        :py:func:`spdl.pipeline._build._build_pipeline` via direct
        attribute assignment to :py:attr:`_underlying_executor` after
        construction. The pool is not known at scheduler construction
        time because it is created inside ``_build_pipeline`` from
        ``num_threads``.
    """

    def __init__(
        self,
        max_concurrent: int,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")

        self._max_concurrent = max_concurrent

        # Bound by _build_pipeline() after construction.
        self._underlying_executor: ThreadPoolExecutor | None = None

        # Per-stage priority lookup, keyed by stage name (StageInfo.stage_name).
        # Populated by register_stage() at build time.
        self._priorities: dict[str, int] = {}

        # Min-heap of (_WorkItem) entries; ordered by (priority, seq).
        self._heap: list[_WorkItem] = []

        # Monotonic sequence number for FIFO tie-breaking. Mutated only on
        # the event-loop thread (via _enqueue from submit()).
        self._seq: int = 0

        # asyncio primitives created lazily inside run() so that the
        # scheduler can be constructed before the event loop exists.
        self._has_work: asyncio.Event | None = None
        self._dispatch_semaphore: asyncio.Semaphore | None = None

    @property
    def max_concurrent(self) -> int:
        """Maximum simultaneous dispatches (read-only)."""
        return self._max_concurrent

    def register_stage(self, info: StageInfo, priority: int) -> None:
        """Register a stage's priority for later dispatch lookup.

        Called during ``_build_node()`` for each sync stage that should
        be routed through the scheduler.

        Args:
            info: The stage's :py:class:`StageInfo` (stage_name is used
                as the registry key).
            priority: Priority value (lower = higher priority; typically
                ``-depth`` so deeper stages dispatch first).
        """
        self._priorities[info.stage_name] = priority
        _LG.debug(
            "PriorityScheduler: registered stage %s with priority=%d",
            info.stage_name,
            priority,
        )

    def get_priority(self, info: StageInfo) -> int:
        """Look up a stage's registered priority. Defaults to 0."""
        return self._priorities.get(info.stage_name, 0)

    def make_stage_executor(self, info: StageInfo) -> "_PrioritizedExecutor":
        """Construct a per-stage :py:class:`_PrioritizedExecutor` shim.

        Called by ``_build_node`` in :py:mod:`spdl.pipeline._components._node`
        when wiring a sync stage. Routing the construction through this
        method keeps ``_node.py`` decoupled from the concrete
        ``_PrioritizedExecutor`` class, avoiding a Buck dep cycle
        between ``spdl/pipeline/_components`` and ``spdl/pipeline``.
        """
        return _PrioritizedExecutor(scheduler=self, info=info)

    # ------------------------------------------------------------------
    # Internal: called from _PrioritizedExecutor on the event-loop thread
    # ------------------------------------------------------------------

    def _make_work_item(
        self,
        *,
        priority: int,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        cf_future: concurrent.futures.Future[Any],
        bridge: dict[str, Any],
    ) -> _WorkItem:
        """Construct a ``_WorkItem``. Caller is on the event-loop thread."""
        self._seq += 1
        return _WorkItem(
            priority=priority,
            seq=self._seq,
            func=func,
            args=args,
            kwargs=kwargs,
            cf_future=cf_future,
            bridge=bridge,
        )

    def _enqueue(self, work_item: _WorkItem) -> None:
        """Push a ``_WorkItem`` onto the heap. On the event-loop thread."""
        heapq.heappush(self._heap, work_item)
        if self._has_work is not None:
            self._has_work.set()

    def _cancel_bridge(self, bridge: dict[str, Any]) -> None:
        """Forward post-dispatch cancellation to the dispatched task.

        Pre-dispatch cancellation is handled by the dispatch loop's
        :py:meth:`~concurrent.futures.Future.set_running_or_notify_cancel`
        check (returns ``False`` for cancelled futures, skipping dispatch).
        Post-dispatch, the underlying :py:class:`ThreadPoolExecutor` work
        cannot be interrupted (per Python docs); we cancel the wrapping
        :py:class:`asyncio.Future` so any awaiters see ``CancelledError``.
        """
        if not bridge["dispatched"]:
            # Pre-dispatch: the dispatch loop will skip this item via
            # set_running_or_notify_cancel(). Nothing to do here.
            return

        task = bridge.get("task")
        if task is not None and not task.done():
            task.cancel()

    # ------------------------------------------------------------------
    # Dispatch loop (runs as a BackgroundTask)
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Dispatch loop. Pops work items in priority order and runs them.

        Runs as a :py:class:`~spdl.pipeline._bg_task.BackgroundTask`
        coroutine on the pipeline's event loop. Exits via
        :py:exc:`asyncio.CancelledError` when the pipeline shuts down.
        """
        if self._underlying_executor is None:
            raise RuntimeError(
                "PriorityScheduler._underlying_executor must be bound by "
                "_build_pipeline() before run() is invoked."
            )

        # Lazy init of asyncio primitives now that we have a running loop.
        has_work = asyncio.Event()
        self._has_work = has_work
        self._dispatch_semaphore = asyncio.Semaphore(self._max_concurrent)
        sem = self._dispatch_semaphore
        loop = asyncio.get_running_loop()

        # If items were enqueued via _PrioritizedExecutor.submit() BEFORE
        # run() was scheduled (e.g., stage tasks fire before the BG task
        # in _run_pipeline_coroutines), they already sit on the heap but
        # _has_work could not have been set (it didn't exist yet). Seed
        # the event so the dispatch loop notices them on the first turn.
        if self._heap:
            has_work.set()

        while True:
            await has_work.wait()
            while self._heap:
                await sem.acquire()
                work_item = heapq.heappop(self._heap)

                cf = work_item.cf_future
                # V5.2: PENDING -> RUNNING here. If pre-dispatch cancel
                # already moved cf to CANCELLED, skip dispatch entirely.
                if not cf.set_running_or_notify_cancel():
                    sem.release()
                    continue

                # Dispatch onto the underlying ThreadPoolExecutor.
                # loop.run_in_executor() schedules the work on the event
                # loop and returns its scheduling Future immediately; we
                # observe completion via add_done_callback rather than
                # awaiting directly. The cast() to object discharges
                # pyre's awaitable-tracking once the callback is wired.
                fut: asyncio.Future[Any] = loop.run_in_executor(
                    self._underlying_executor,
                    self._invoke,
                    work_item,
                )
                fut.add_done_callback(
                    lambda f, wi=work_item, s=sem: self._on_complete(f, wi, s)
                )
                work_item.bridge["task"] = cast(object, fut)
                work_item.bridge["dispatched"] = True
            has_work.clear()

    @staticmethod
    def _invoke(work_item: _WorkItem) -> Any:
        """Worker-thread entrypoint. Runs the user's function."""
        return work_item.func(*work_item.args, **work_item.kwargs)

    def _on_complete(
        self,
        fut: asyncio.Future[Any],
        work_item: _WorkItem,
        sem: asyncio.Semaphore,
    ) -> None:
        """Bridge completion back to the caller's ``cf_future``.

        Runs on the event-loop thread (asyncio future done-callback).
        """
        sem.release()
        cf = work_item.cf_future
        # cf is in RUNNING at this point (set at dispatch). Final state
        # is FINISHED (set_result/set_exception). asyncio.CancelledError
        # raised by the underlying task is surfaced as an exception on cf.
        if fut.cancelled():
            cf.set_exception(asyncio.CancelledError())
            return
        exc = fut.exception()
        if exc is not None:
            cf.set_exception(exc)
        else:
            cf.set_result(fut.result())

    def _drain_pending(self) -> None:
        """Cancel everything still on the heap on shutdown.

        Called from :py:class:`_PrioritySchedulerBackgroundTask` after
        the dispatch loop exits. Items still on the heap have
        ``cf_future`` in PENDING state; cancelling them lets any awaiting
        :py:meth:`asyncio.AbstractEventLoop.run_in_executor` callers see
        :py:exc:`asyncio.CancelledError` cleanly.
        """
        while self._heap:
            work_item = heapq.heappop(self._heap)
            cf = work_item.cf_future
            cf.cancel()


class _PrioritizedExecutor(Executor):
    """Thin :py:class:`Executor` adapter that submits work through
    :py:class:`PriorityScheduler`.

    One instance per sync stage — captures the stage's
    :py:class:`StageInfo` so the scheduler can look up the right priority.

    NOT a real :py:class:`ThreadPoolExecutor` — it has no threads of its
    own. :py:meth:`submit` enqueues into the scheduler's heap, which
    dispatches to the real shared pool when capacity is available.

    Threading model (V5.3): :py:meth:`submit` is called by
    :py:meth:`asyncio.AbstractEventLoop.run_in_executor` synchronously on
    the pipeline's event-loop thread. We therefore can fetch the loop
    via :py:func:`asyncio.get_running_loop` *per-call* — no pre-stamping
    is required, which avoids races during pipeline startup before the
    BG task has begun.

    Cancellation: post-dispatch ``cf_future.cancel()`` returns ``False``
    (matching :py:class:`ThreadPoolExecutor` semantics). Pre-dispatch
    cancel works correctly via
    :py:meth:`~concurrent.futures.Future.set_running_or_notify_cancel`
    being called at dispatch time.
    """

    def __init__(
        self,
        scheduler: PriorityScheduler,
        info: StageInfo,
    ) -> None:
        self._scheduler = scheduler
        self._info = info

    # pyre-ignore[14]: Executor.submit's typeshed stub uses an internal
    # TypeVar that cannot be matched from a subclass override; pyre flags
    # any concrete signature here as Inconsistent override even when the
    # runtime semantics are identical to the base class. This is the same
    # workaround used by other Executor subclasses (the alternative is to
    # restructure the entire dispatch API).
    def submit(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> concurrent.futures.Future[Any]:
        # V5.3: per-call loop fetch. Works because loop.run_in_executor()
        # invokes executor.submit() synchronously on the loop thread.
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()

        cf_future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        # cf_future is in PENDING state. set_running_or_notify_cancel() is
        # called at DISPATCH time (in PriorityScheduler.run()), NOT here.

        bridge: dict[str, Any] = {
            "task": None,  # asyncio.Future from run_in_executor; set after dispatch
            "dispatched": False,  # True once the asyncio task starts
        }

        def _enqueue() -> None:
            # Runs on the loop thread.
            work_item = self._scheduler._make_work_item(
                priority=self._scheduler.get_priority(self._info),
                func=fn,
                args=args,
                kwargs=kwargs,
                cf_future=cf_future,
                bridge=bridge,
            )
            self._scheduler._enqueue(work_item)

        # call_soon_threadsafe is correct even when caller is the loop
        # thread (defensive; also keeps submit() safe from non-loop threads
        # such as unit tests).
        loop.call_soon_threadsafe(_enqueue)

        def _on_cancel(cf_fut: concurrent.futures.Future[Any]) -> None:
            # Called by concurrent.futures when cf_future transitions to
            # a done state. We only act on the cancellation case.
            if not cf_fut.cancelled():
                return
            loop.call_soon_threadsafe(self._scheduler._cancel_bridge, bridge)

        cf_future.add_done_callback(_on_cancel)
        return cf_future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """No-op: the underlying scheduler owns the worker pool lifecycle."""
        # The shared ThreadPoolExecutor is owned by the Pipeline, not by
        # this shim. Shutdown happens via Pipeline.stop() and the BG task
        # cancellation, which triggers _drain_pending() on the scheduler.
        return None


class _PrioritySchedulerBackgroundTask(BackgroundTask):
    """:py:class:`BackgroundTask` adapter that runs
    :py:meth:`PriorityScheduler.run` on the pipeline's event loop.

    Lifecycle (from :py:class:`spdl.pipeline._bg_task.BackgroundTask`):

    - Started by ``_run_pipeline_coroutines()`` AFTER stage tasks are
      created.
    - Cancelled when ALL stage tasks complete (or on pipeline failure).
    - Errors are logged, do NOT fail the pipeline.

    On cancellation, ensures any items left on the heap are cancelled so
    that awaiting :py:meth:`asyncio.AbstractEventLoop.run_in_executor`
    callers don't hang.
    """

    def __init__(self, scheduler: PriorityScheduler) -> None:
        self._scheduler = scheduler

    async def run(self) -> None:
        try:
            await self._scheduler.run()
        finally:
            self._scheduler._drain_pending()
