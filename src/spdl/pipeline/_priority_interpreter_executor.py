# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Priority-queue-based InterpreterPoolExecutor for SPDL pipelines.

Python 3.14+ only. Provides an InterpreterPoolExecutor variant that
prioritizes downstream pipeline stages over upstream ones.

Usage::

    pool = PriorityInterpreterPoolExecutor(max_workers=4)

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
stages should be created after upstream ones.
"""

from __future__ import annotations

import sys
from typing import Any, Callable

from ._priority_executor import (
    _assign_owner_id,
    _OWNER_REGISTRY,
    _PriorityExecutorBase,
    _PriorityQueueAdapter,
    PriorityExecutorEntrypoint,
)

__all__ = [
    "PriorityInterpreterPoolExecutor",
]


if sys.version_info < (3, 14):
    from concurrent.futures import Executor

    class PriorityInterpreterPoolExecutor(Executor):
        """Stub for Python < 3.14. Raises ``RuntimeError`` on instantiation."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                f"PriorityInterpreterPoolExecutor requires Python 3.14 or later. "
                f"Current version: {sys.version_info.major}.{sys.version_info.minor}"
            )

        def get_executor(
            self, *, priority: int | None = None
        ) -> PriorityExecutorEntrypoint:
            raise RuntimeError(
                "PriorityInterpreterPoolExecutor requires Python 3.14 or later."
            )

else:
    from concurrent.futures import (
        _base,  # pyre-ignore[21]
        Future,
        thread as _thread_mod,  # pyre-ignore[21]
    )
    from concurrent.futures.interpreter import (  # pyre-ignore[21]
        BrokenInterpreterPool,
        InterpreterPoolExecutor,
    )
    from concurrent.futures.thread import _WorkItem  # pyre-ignore[21]

    class PriorityInterpreterPoolExecutor(_PriorityExecutorBase):  # type: ignore[no-redef]
        """An ``InterpreterPoolExecutor`` wrapper with priority-ordered queue.

        Use ``get_executor()`` to create per-stage executors. Executors
        created later automatically have higher priority (processed first).

        Supports the pickle protocol for use with
        ``run_pipeline_in_subprocess()``.

        Python 3.14+ only.

        Example::

            pool = PriorityInterpreterPoolExecutor(max_workers=4)

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

        _pool_executor_class: type = InterpreterPoolExecutor

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
            self._pool: InterpreterPoolExecutor = InterpreterPoolExecutor(
                **self._kwargs
            )
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
                    raise BrokenInterpreterPool(pool._broken)
                if pool._shutdown:
                    raise RuntimeError("cannot schedule new futures after shutdown")
                if _thread_mod._shutdown:
                    raise RuntimeError(
                        "cannot schedule new futures after interpreter shutdown"
                    )

                f: Future[Any] = _base.Future()
                task = pool._resolve_work_item_task(fn, args, kwargs)
                w = _WorkItem(f, task)
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
            self._pool = InterpreterPoolExecutor(**self._kwargs)
            self._work_queue = _PriorityQueueAdapter()
            self._pool._work_queue = self._work_queue  # type: ignore[assignment]
            _OWNER_REGISTRY[self._id] = self
