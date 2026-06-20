# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PathVariants support: fan-out routing and fan-in merging within a pipeline."""

# pyre-strict

__all__ = [
    "_path_variants_router",
]

import asyncio
import inspect
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Sequence
from contextlib import asynccontextmanager, AsyncExitStack
from typing import Any

from spdl.pipeline._common._convert import _to_async

from ._common import _EOF, _EPOCH_END, is_eof, is_epoch_end
from ._hook import _stage_hooks, TaskHook
from ._queue import AsyncQueue

_LG: logging.Logger = logging.getLogger(__name__)


def _make_async_router(
    router: Callable[[Any], int] | Callable[[Any], Awaitable[int]],
) -> Callable[[Any], Awaitable[int]]:
    """Wrap a sync router with run_in_executor; pass through async routers."""
    if inspect.iscoroutinefunction(router):
        return router
    call = getattr(router, "__call__", None)
    if call is not None and inspect.iscoroutinefunction(call):
        return router  # pyre-ignore[7]
    return _to_async(router, executor=None)  # pyre-ignore[7]


@asynccontextmanager
async def _queue_stage_hook(queues: Sequence[AsyncQueue]) -> AsyncGenerator[None, None]:
    # Custom _queue_stage_hook.
    # In addition to the _queue_stage_hook used by other pipes, this class has
    # - Supports variable number of queues
    # - Put _EOF even when the task is cancelled, so as to handle the case
    #   where the cancellation is originated from one of the variant paths.
    #   If one of the variant path cause the cancellation of upstream tasks,
    #   we need to stop the other variant paths, which is not handled by the
    #   general cancellation mechanism. To handle this, we always put _EOF.
    #   When doing this, to handle the case where the queue is full,
    #   we put trial-and-error, and make sure that the _EOF is processed.
    cancelled = False
    async with AsyncExitStack() as stack:
        for q in queues:
            await stack.enter_async_context(q.stage_hook())
        try:
            yield
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            for i, q in enumerate(queues):
                try:
                    if cancelled:
                        # When cancelled (e.g. a variant path failed), we cannot
                        # await, so evict items to make room for EOF.
                        while q.full():
                            q.get_nowait()
                        q.put_nowait(_EOF)
                    else:
                        # Normal shutdown: wait for queue space so we don't
                        # drop items that downstream hasn't consumed yet.
                        await q.put(_EOF)
                except Exception:
                    _LG.error(
                        "Failed to pass EOF to path:%d. The pipeline might not shutdown properly.",
                        i,
                    )


def _path_variants_router(
    input_queue: AsyncQueue,
    path_queues: Sequence[AsyncQueue],
    router: Callable[[Any], int] | Callable[[Any], Awaitable[int]],
    task_hooks: list[TaskHook],
) -> Coroutine[None, None, None]:
    """Create a coroutine that routes items to per-path queues.

    The router reads items from ``input_queue``, calls ``router(item)`` to
    determine the target path index, and puts the item on the corresponding
    queue in ``path_queues``.

    Sync routers are wrapped with ``run_in_executor`` to avoid blocking the
    event loop. Async routers are awaited directly.

    On completion (EOF or error), the router sends EOF to ALL path queues in
    a ``finally`` block so that every path shuts down cleanly.

    The epoch-end sentinel (emitted by a continuous source between epochs) is
    not a routable item, so it is broadcast to ALL path queues rather than
    passed to ``router``. This lets each path flush its in-flight work and lets
    the downstream fan-in merge observe the epoch boundary on every input queue.

    Args:
        input_queue: The queue to consume items from.
        path_queues: Per-path output queues, one per variant path.
        router: Callable that maps an item to a path index.
        task_hooks: Hooks for monitoring.

    Returns:
        A coroutine that executes the router stage.
    """
    num_paths: int = len(path_queues)
    arouter: Callable[[Any], Awaitable[int]] = _make_async_router(router)

    @_queue_stage_hook(path_queues)
    # pyrefly: ignore [not-callable]
    @_stage_hooks(task_hooks)
    async def _router() -> None:
        while True:
            item = await input_queue.get()
            if is_eof(item):
                return

            if is_epoch_end(item):
                for q in path_queues:
                    await q.put(_EPOCH_END)
                continue

            idx = await arouter(item)
            if idx < 0 or idx >= num_paths:
                raise IndexError(
                    f"Router returned index {idx}, but there are only "
                    f"{num_paths} paths (valid range: [0, {num_paths}))."
                )
            await path_queues[idx].put(item)

    return _router()
