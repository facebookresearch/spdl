# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PathVariants support: fan-out routing and fan-in merging within a pipeline."""

# pyre-strict

__all__ = [
    "_batched_path_variants_merge",
    "_path_variants_router",
]

import asyncio
import inspect
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Sequence
from contextlib import asynccontextmanager, AsyncExitStack
from typing import Any

from spdl.pipeline._common._convert import _to_async

from ._common import _EOF, _EPOCH_END, _ShieldedHook, is_eof, is_epoch_end, StageInfo
from ._hook import _stage_hooks, TaskHook
from ._queue import AsyncQueue

_LG: logging.Logger = logging.getLogger(__name__)


def _make_async_router(
    router: Callable[[Any], Any] | Callable[[Any], Awaitable[Any]],
) -> Callable[[Any], Awaitable[Any]]:
    """Wrap a sync router with run_in_executor; pass through async routers.

    The router returns an ``int`` (per-item mode) or a ``Sequence[int]`` (batched
    mode); this wrapper is agnostic to which."""
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
            # Use the shared guarded wrapper so stage finalization (e.g. final
            # stats flush) survives cancellation, consistently with the other
            # pipes. This stage's only bespoke behavior is the multi-queue EOF
            # broadcast below, which the shared hook does not cover.
            await stack.enter_async_context(_ShieldedHook(q.stage_hook()))
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
    router: Callable[[Any], Any] | Callable[[Any], Awaitable[Any]],
    task_hooks: list[TaskHook],
    batched: bool = False,
) -> Coroutine[None, None, None]:
    """Create a coroutine that routes items to per-path queues.

    In the default (per-item) mode the router reads items from ``input_queue``,
    calls ``router(item)`` to determine the target path index, and puts the item
    on the corresponding queue in ``path_queues``.

    In ``batched`` mode each item read from ``input_queue`` is a whole batch (a
    list). ``router(batch)`` returns one path index per element; the batch is
    partitioned into per-path sub-batches (preserving order) and each sub-batch
    is put on its path's queue as a single list. A sub-batch is put on **every**
    path queue -- an empty list where a path received no elements -- so that each
    input batch contributes exactly one list to every path queue. This keeps the
    downstream fan-in merge in lockstep, letting it recombine the sub-batches of
    one input batch by reading one list from each path (see
    :py:func:`_batched_path_variants_merge`).

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
        router: Callable that maps an item (or a batch, if ``batched``) to a path
            index (or a sequence of per-element path indices, if ``batched``).
        task_hooks: Hooks for monitoring.
        batched: If True, route whole batches (see above) instead of single items.

    Returns:
        A coroutine that executes the router stage.
    """
    num_paths: int = len(path_queues)
    arouter: Callable[[Any], Awaitable[Any]] = _make_async_router(router)

    async def _route_item(item: Any) -> None:
        idx = await arouter(item)
        if idx < 0 or idx >= num_paths:
            raise IndexError(
                f"Router returned index {idx}, but there are only "
                f"{num_paths} paths (valid range: [0, {num_paths}))."
            )
        await path_queues[idx].put(item)

    async def _route_batch(batch: Any) -> None:
        indices = await arouter(batch)
        if len(indices) != len(batch):
            raise ValueError(
                f"Batched router returned {len(indices)} indices for a batch of "
                f"{len(batch)} items; it must return exactly one index per item."
            )
        parts: list[list[Any]] = [[] for _ in range(num_paths)]
        for item, idx in zip(batch, indices):
            if idx < 0 or idx >= num_paths:
                raise IndexError(
                    f"Router returned index {idx}, but there are only "
                    f"{num_paths} paths (valid range: [0, {num_paths}))."
                )
            parts[idx].append(item)
        # Emit to every path (empty lists included) so each input batch contributes
        # exactly one list per path, keeping the fan-in merge in lockstep.
        for q, part in zip(path_queues, parts):
            await q.put(part)

    _route: Callable[[Any], Awaitable[None]] = _route_batch if batched else _route_item

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

            await _route(item)

    return _router()


async def _batched_path_variants_merge(
    info: StageInfo,
    input_queues: Sequence[asyncio.Queue],
    output_queue: asyncio.Queue,
) -> None:
    """Fan-in merge for a ``batched`` path_variants stage.

    Each input batch was partitioned by the router into one sub-batch (list) per
    path, so the queues stay in lockstep: reading one list from every input queue
    yields the sub-batches of a single original batch. They are concatenated (in
    path order) back into one batch and emitted, so the stage is batch-in /
    batch-out and downstream sees whole batches -- not the individual items.

    A fully-empty recombined batch (every path routed nothing, or every element
    was dropped) is skipped rather than emitted downstream.

    ``_EPOCH_END`` and ``_EOF`` are broadcast by the router to every path queue,
    so they surface on all input queues together; the merge forwards one epoch
    boundary and stops on end-of-stream.
    """
    while True:
        items = [await q.get() for q in input_queues]
        if any(is_eof(it) for it in items):
            return
        if any(is_epoch_end(it) for it in items):
            await output_queue.put(_EPOCH_END)
            continue
        combined = [item for sub_batch in items for item in sub_batch]
        if combined:
            await output_queue.put(combined)
