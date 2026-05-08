# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Priority-queue-based executor for SPDL pipelines.

Provides executors that prioritize downstream pipeline stages over upstream ones
when sharing a thread/process pool. This reduces end-to-end latency by ensuring
items that are closer to the pipeline output get processed first.

Usage::

    pool = PriorityThreadPoolExecutor(max_workers=4)

    pipeline = (
        PipelineBuilder()
        .add_source(range(100))
        .pipe(load, executor=pool.get_executor(), concurrency=4)
        .pipe(decode, executor=pool.get_executor(), concurrency=4)
        .pipe(transform, executor=pool.get_executor(), concurrency=4)
        .add_sink(3)
        .build(num_threads=1)
    )

Executors created later automatically have higher priority. Downstream
stages should be created after upstream ones. Priority can be overridden
explicitly via ``get_executor(priority=N)``.

Both the pool executor and ``PriorityExecutorEntrypoint`` support the
pickle protocol, so they can be used with ``run_pipeline_in_subprocess()``.
"""

from __future__ import annotations

import itertools
import sys
import weakref
from collections.abc import Callable
from concurrent.futures import (
    _base,  # pyre-ignore[21]
    Executor,
    Future,
    process as _process_mod,  # pyre-ignore[21]
    ProcessPoolExecutor,
    thread as _thread_mod,  # pyre-ignore[21]
    ThreadPoolExecutor,
)
from concurrent.futures.process import (  # pyre-ignore[21]
    _WorkItem as _ProcessWorkItem,
    BrokenProcessPool,
)
from concurrent.futures.thread import (  # pyre-ignore[21]
    _WorkItem as _ThreadWorkItem,
    BrokenThreadPool,
)
from queue import PriorityQueue
from typing import Any, Self

__all__ = [
    "PriorityExecutorEntrypoint",
    "PriorityProcessPoolExecutor",
    "PriorityThreadPoolExecutor",
]

# ---------------------------------------------------------------------------
# Global registry for reconnecting entrypoints to their owner after pickle.
# Uses weak references so owners can be garbage-collected when no entrypoints
# or user code reference them. Each process has its own registry.
# ---------------------------------------------------------------------------

_OWNER_REGISTRY: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()
_next_owner_id = 0


def _assign_owner_id() -> int:
    global _next_owner_id
    oid = _next_owner_id
    _next_owner_id += 1
    return oid


class _PriorityQueueAdapter:
    """Drop-in replacement for SimpleQueue/Queue that orders items by priority.

    When ``put`` is called without a priority (e.g. shutdown sentinels injected
    by the base executor), items receive the highest priority so they are
    processed immediately.
    """

    def __init__(self) -> None:
        self._queue: PriorityQueue[tuple[tuple[float, ...], int, Any]] = PriorityQueue()
        self._seq: itertools.count[int] = itertools.count()

    def put(self, item: Any, priority: tuple[int, int] | None = None) -> None:
        if priority is None:
            p: tuple[float, ...] = (float("-inf"),)
        else:
            p = priority
        self._queue.put((p, next(self._seq), item))

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        _, _, item = self._queue.get(block=block, timeout=timeout)
        return item

    def get_nowait(self) -> Any:
        _, _, item = self._queue.get_nowait()
        return item

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()


# ---------------------------------------------------------------------------
# Base class for priority pool executors
# ---------------------------------------------------------------------------


class _PriorityExecutorBase:
    """Shared logic for priority-queue pool executors."""

    _kwargs: dict[str, Any]  # pyre-ignore[13]
    _priority_counter: int  # pyre-ignore[13]
    _id: int  # pyre-ignore[13]

    @property
    def _max_workers(self) -> int | None:
        return self._kwargs["max_workers"]

    def _submit_with_priority(
        self,
        priority: tuple[int, int],
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Future[Any]:
        raise NotImplementedError

    def get_executor(
        self, *, priority: int | None = None
    ) -> PriorityExecutorEntrypoint:
        """Create a child executor for a pipeline stage.

        Args:
            priority: Priority level. Higher = processed first. When
                omitted, auto-assigns an incrementing priority so that
                executors created later have higher priority.

        Returns:
            A ``PriorityExecutorEntrypoint`` that can be passed to
            ``PipelineBuilder.pipe()``.
        """
        if priority is None:
            self._priority_counter -= 1
            internal_priority = self._priority_counter
        else:
            internal_priority = -priority
        return PriorityExecutorEntrypoint(self, internal_priority)

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        raise NotImplementedError

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.shutdown(wait=True)

    def __getstate__(self) -> dict[str, Any]:
        return {
            "id": self._id,
            "kwargs": self._kwargs,
            "priority_counter": self._priority_counter,
        }


# ---------------------------------------------------------------------------
# Pool executors — composition-based, picklable
# ---------------------------------------------------------------------------


class PriorityThreadPoolExecutor(_PriorityExecutorBase):
    """A ``ThreadPoolExecutor`` wrapper whose internal queue is priority-ordered.

    Use ``get_executor()`` to create per-stage executors. Executors
    created later automatically have higher priority (processed first).
    Priority can be overridden explicitly via ``get_executor(priority=N)``.

    Supports the pickle protocol for use with ``run_pipeline_in_subprocess()``.

    Example::

        pool = PriorityThreadPoolExecutor(max_workers=4)

        pipeline = (
            PipelineBuilder()
            .add_source(range(100))
            .pipe(load, executor=pool.get_executor(), concurrency=4)
            .pipe(decode, executor=pool.get_executor(), concurrency=4)
            .pipe(transform, executor=pool.get_executor(), concurrency=4)
            .add_sink(3)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            for item in pipeline.get_iterator():
                process(item)

        pool.shutdown()
    """

    _pool_executor_class: type[ThreadPoolExecutor] = ThreadPoolExecutor

    def __init__(
        self,
        max_workers: int | None = None,
        thread_name_prefix: str = "",
        initializer: Callable[..., object] | None = None,
        initargs: tuple[Any, ...] = (),
    ) -> None:
        self._kwargs: dict[str, Any] = {
            "max_workers": max_workers,
            "thread_name_prefix": thread_name_prefix,
            "initializer": initializer,
            "initargs": initargs,
        }
        self._pool: ThreadPoolExecutor = ThreadPoolExecutor(**self._kwargs)
        self._work_queue: _PriorityQueueAdapter = _PriorityQueueAdapter()
        self._pool._work_queue = self._work_queue  # type: ignore[assignment]
        self._priority_counter: int = 0
        self._id: int = _assign_owner_id()
        _OWNER_REGISTRY[self._id] = self

    def _submit_with_priority(
        self,
        priority: tuple[int, int],
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Future[Any]:
        pool = self._pool
        with pool._shutdown_lock, _thread_mod._global_shutdown_lock:
            if pool._broken:
                raise BrokenThreadPool(pool._broken)
            if pool._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            if _thread_mod._shutdown:
                raise RuntimeError(
                    "cannot schedule new futures after interpreter shutdown"
                )

            f: Future[Any] = _base.Future()
            if hasattr(pool, "_resolve_work_item_task"):
                task = pool._resolve_work_item_task(fn, args, kwargs)  # pyre-ignore[16]
                w = _ThreadWorkItem(f, task)  # pyre-ignore[20]
            else:
                w = _ThreadWorkItem(f, fn, args, kwargs)
            self._work_queue.put(w, priority=priority)
            pool._adjust_thread_count()
            return f

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        pool = self._pool
        with pool._shutdown_lock:
            pool._shutdown = True
            if cancel_futures:
                while not self._work_queue.empty():
                    try:
                        work_item = self._work_queue.get_nowait()
                    except Exception:
                        break
                    if work_item is not None:
                        work_item.future.cancel()
                        work_item.future.set_running_or_notify_cancel()
            for _ in pool._threads:
                self._work_queue.put(None, priority=(sys.maxsize, 0))
        if wait:
            for t in pool._threads:
                t.join()
        _OWNER_REGISTRY.pop(self._id, None)

    def __setstate__(self, state: dict[str, Any]) -> None:
        existing = _OWNER_REGISTRY.get(state["id"])
        if existing is not None:
            self.__dict__.update(existing.__dict__)
            return
        self._kwargs = state["kwargs"]
        self._priority_counter = state["priority_counter"]
        self._id = state["id"]
        self._pool = ThreadPoolExecutor(**self._kwargs)
        self._work_queue = _PriorityQueueAdapter()
        self._pool._work_queue = self._work_queue  # type: ignore[assignment]
        _OWNER_REGISTRY[self._id] = self


class PriorityProcessPoolExecutor(_PriorityExecutorBase):
    """A ``ProcessPoolExecutor`` wrapper whose work-ID queue is priority-ordered.

    The ``_work_ids`` queue determines which pending work items are fed
    to worker processes first. Replacing it with a priority queue ensures
    higher-priority items are dispatched first.

    Supports the pickle protocol for use with ``run_pipeline_in_subprocess()``.

    Example::

        pool = PriorityProcessPoolExecutor(max_workers=4)

        pipeline = (
            PipelineBuilder()
            .add_source(range(100))
            .pipe(cpu_bound_load, executor=pool.get_executor(), concurrency=4)
            .pipe(cpu_bound_decode, executor=pool.get_executor(), concurrency=4)
            .add_sink(3)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            for item in pipeline.get_iterator():
                process(item)

        pool.shutdown()
    """

    _pool_executor_class: type[ProcessPoolExecutor] = ProcessPoolExecutor

    def __init__(
        self,
        max_workers: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._kwargs: dict[str, Any] = {"max_workers": max_workers, **kwargs}
        self._pool: ProcessPoolExecutor = ProcessPoolExecutor(**self._kwargs)
        self._work_queue: _PriorityQueueAdapter = _PriorityQueueAdapter()
        self._pool._work_ids = self._work_queue  # type: ignore[assignment]
        self._priority_counter: int = 0
        self._id: int = _assign_owner_id()
        _OWNER_REGISTRY[self._id] = self

    def _submit_with_priority(
        self,
        priority: tuple[int, int],
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Future[Any]:
        pool = self._pool
        with pool._shutdown_lock:
            if pool._broken:
                raise BrokenProcessPool(pool._broken)
            if pool._shutdown_thread:
                raise RuntimeError("cannot schedule new futures after shutdown")
            if _process_mod._global_shutdown:
                raise RuntimeError(
                    "cannot schedule new futures after interpreter shutdown"
                )

            f: Future[Any] = _base.Future()
            w = _ProcessWorkItem(f, fn, args, kwargs)

            pool._pending_work_items[pool._queue_count] = w
            self._work_queue.put(pool._queue_count, priority=priority)
            pool._queue_count += 1
            if hasattr(pool, "_executor_manager_thread_wakeup"):
                pool._executor_manager_thread_wakeup.wakeup()  # pyre-ignore[16]
            elif hasattr(pool, "_result_queue"):
                pool._result_queue.put(None)  # pyre-ignore[16]

            if getattr(
                pool, "_safe_to_dynamically_spawn_children", False
            ):  # pyre-ignore[16]
                pool._adjust_process_count()  # pyre-ignore[16]
            pool._start_executor_manager_thread()
            return f

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)
        _OWNER_REGISTRY.pop(self._id, None)

    def __setstate__(self, state: dict[str, Any]) -> None:
        existing = _OWNER_REGISTRY.get(state["id"])
        if existing is not None:
            self.__dict__.update(existing.__dict__)
            return
        self._kwargs = state["kwargs"]
        self._priority_counter = state["priority_counter"]
        self._id = state["id"]
        self._pool = ProcessPoolExecutor(**self._kwargs)
        self._work_queue = _PriorityQueueAdapter()
        self._pool._work_ids = self._work_queue  # type: ignore[assignment]
        _OWNER_REGISTRY[self._id] = self


# ---------------------------------------------------------------------------
# Per-stage entrypoint — picklable, reconnects to owner via registry
# ---------------------------------------------------------------------------


class PriorityExecutorEntrypoint(Executor):
    """Per-stage executor proxy with a fixed priority.

    Drop-in compatible with ``concurrent.futures.Executor``.
    Created via ``PriorityThreadPoolExecutor.get_executor()`` or
    ``PriorityProcessPoolExecutor.get_executor()``.

    Supports the pickle protocol: on deserialization, the first entrypoint
    with a given owner ID creates the owner; subsequent ones reuse it.
    """

    def __init__(
        self,
        owner: _PriorityExecutorBase,
        priority: int,
    ) -> None:
        self._owner_id: int = owner._id
        self._owner_class: type[_PriorityExecutorBase] = type(owner)
        self._owner_kwargs: dict[str, Any] = owner._kwargs
        self._priority: int = priority
        self._counter: int = 0
        self._owner_ref: _PriorityExecutorBase = owner

    @property
    def _owner(self) -> _PriorityExecutorBase:
        owner = self.__dict__.get("_owner_ref")
        if owner is not None:
            return owner
        owner = _OWNER_REGISTRY.get(self._owner_id)
        if owner is None:
            owner = self._owner_class(**self._owner_kwargs)
            old_id = owner._id
            _OWNER_REGISTRY.pop(old_id, None)
            owner._id = self._owner_id
            _OWNER_REGISTRY[self._owner_id] = owner
        self._owner_ref = owner
        return owner

    def submit(  # pyre-ignore[14]
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Future[Any]:
        priority = (self._priority, self._counter)
        self._counter += 1
        return self._owner._submit_with_priority(
            priority,
            fn,
            args,
            kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._owner, name)

    def __getstate__(self) -> dict[str, Any]:
        return {
            "owner_id": self._owner_id,
            "owner_class": self._owner_class,
            "owner_kwargs": self._owner_kwargs,
            "priority": self._priority,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._owner_id = state["owner_id"]
        self._owner_class = state["owner_class"]
        self._owner_kwargs = state["owner_kwargs"]
        self._priority = state["priority"]
        self._counter = 0
