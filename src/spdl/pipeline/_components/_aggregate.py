# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "_aggregate",
]

import asyncio
from collections.abc import Coroutine
from typing import Any

from spdl.pipeline.defs import Aggregator

from ._common import _SKIP, is_eof
from ._hook import _stage_hooks, _task_hooks, TaskHook
from ._pipe import _FailCounter
from ._queue import _queue_stage_hook, AsyncQueue

# pyre-strict


class _AggregatorWrapper:
    def __init__(self, agg: Aggregator) -> None:
        self._agg = agg

    def __call__(self, item: Any | None) -> Any | None:
        if is_eof(item):
            return self._agg.flush()
        else:
            return self._agg.accumulate(item)


def _aggregate(
    name: str,
    input_queue: AsyncQueue,
    output_queue: AsyncQueue,
    op: Aggregator,
    fail_counter: _FailCounter,
    task_hooks: list[TaskHook],
    op_requires_eof: bool,
) -> Coroutine:
    """Create a coroutine for aggregation that drains the input queue in bulk.

    Unlike ``_pipe`` which consumes one item per event loop iteration and creates
    an asyncio Task for each item, this function drains all available items from
    the input queue after the initial blocking get, and processes them synchronously.
    This reduces context switching overhead for aggregation operations where the
    per-item work (accumulate) is trivial.

    Args:
        name: The name of the pipeline stage for logging and task naming.
        input_queue: The queue to consume input items from.
        output_queue: The queue to put aggregated items into.
        op: The aggregator instance implementing the Aggregator protocol.
        fail_counter: Hook for tracking and limiting task failures.
        task_hooks: List of hooks for monitoring task execution.
        op_requires_eof: If True, pass EOF token to the operation; otherwise stop
            processing before EOF.

    Returns:
        A coroutine that executes the aggregation stage.

    Raises:
        ValueError: If input_queue and output_queue are the same object.
        RuntimeError: If the number of failures exceeds the threshold.
    """
    if input_queue is output_queue:
        raise ValueError("input queue and output queue must be different.")

    hooks: list[TaskHook] = [*task_hooks, fail_counter]
    wrapper: _AggregatorWrapper = _AggregatorWrapper(op)

    @_queue_stage_hook(output_queue)
    @_stage_hooks(hooks)
    async def aggregate_pipe() -> None:
        while not fail_counter.too_many_failures():
            # Block until at least one item is available
            item = await input_queue.get()

            # Process each item immediately. If the aggregator emits
            # something, stop draining and yield control so downstream
            # can consume the result. Otherwise, keep draining without
            # blocking to reduce context switching overhead.
            while True:
                if is_eof(item) and not op_requires_eof:
                    return

                async with _task_hooks(hooks, item):
                    result = wrapper(item)

                if result is not _SKIP:
                    await output_queue.put(result)

                if is_eof(item):
                    return
                elif result is not _SKIP:
                    break  # Aggregator emitted, stop draining

                # Aggregator didn't emit, try to drain next item
                try:
                    item = input_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        if fail_counter.too_many_failures():
            fail_counter.raise_for_failures(name)

    return aggregate_pipe()
