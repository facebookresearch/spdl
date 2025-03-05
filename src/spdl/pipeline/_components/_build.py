# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "_build_pipeline",
    "PipelineFailure",
    "_SourceConfig",
    "_ProcessConfig",
    "_SinkConfig",
]

import asyncio
import enum
from collections.abc import (
    AsyncIterable,
    Coroutine,
    Iterable,
)
from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, TypeVar

from .._queue import AsyncQueue, StatsQueue as DefaultQueue
from .._utils import create_task
from ._pipe import (
    _ordered_pipe,
    _pipe,
    _PipeArgs,
)
from ._sink import _sink
from ._source import _source

# pyre-strict

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class _SourceConfig(Generic[T]):
    source: Iterable | AsyncIterable
    queue_class: type[AsyncQueue[T]] | None


class _PType(enum.IntEnum):
    Pipe = 1
    OrderedPipe = 2
    Aggregate = 3
    Disaggregate = 4


@dataclass
class _ProcessConfig(Generic[T, U]):
    type_: _PType
    args: _PipeArgs[T, U]
    queue_class: type[AsyncQueue[U]] | None


@dataclass
class _SinkConfig(Generic[T]):
    buffer_size: int
    queue_class: type[AsyncQueue[T]] | None


# The following is how we intend pipeline to behave. If the implementation
# is inconsistent with the following, it is a bug.
#
# 1. Successful execution.
#
#    Assumption: The consumer keeps consuming the data from the output queue.
#
#    Each stage completes without failure.
#    The source will pass EOF, and each stage receiving EOF will shut down,
#    then pass EOF.
#    EOF is not propagated to the output queue.
#
# 2. Failures at some stage. One or more stages fail.
#
#    Assumption: The consumer keeps consuming the data from the output queue.
#
#    Resolution:
#      We want to keep the tasks downstream to failed ones alive, so that they
#      can finish the ongoing works.
#
#    Actions:
#      - The stage must pass EOF to the next queue, so that downstream stages
#        can exit cleanly.
#      - Passing EOF to the next queue can be blocked if the queue is full.
#      - For a graceful exit, the assumption must be met.
#      - The stages upstream to the failure must be cancelled.
#
# 3. Cancelled: The pipeline execution is cancelled.
#
#    Assumption: The consumer stopped consuming the data from the output queue.
#
#    Actions:
#      - All the stages must be cancelled.
#        (If tasks are blocked on async await, cancelling should do).
#
# Following the above intention, each stage must pass EOF to the output queue
# unless the task was cancelled. See `_queue_stage_hook`.
#


def _get_queue(
    queue_class: type[AsyncQueue[T]] | None,
    interval: float,
) -> type[AsyncQueue[T]]:
    if queue_class is not None:
        return queue_class

    return partial(DefaultQueue, interval=interval)  # pyre-ignore: [7]


def _build_pipeline(
    src: _SourceConfig[T],
    process_args: list[_ProcessConfig[Any, Any]],  # pyre-ignore: [2]
    sink: _SinkConfig[U],
    report_stats_interval: float,
) -> tuple[Coroutine[None, None, None], AsyncQueue[U]]:
    # Note:
    # Make sure that coroutines are ordered from source to sink.
    # `_run_pipeline_coroutines` expects and rely on this ordering.
    coros = []
    queues = []

    # source
    queue_class = _get_queue(src.queue_class, report_stats_interval)
    queues.append(queue_class("0:src_queue", 1))
    coros.append(("Pipeline::0:src", _source(src.source, queues[0])))

    # pipes
    for i, cfg in enumerate(process_args, start=1):
        queue_class = _get_queue(cfg.queue_class, report_stats_interval)
        queue_name = f"{cfg.args.name}_queue"
        queues.append(queue_class(queue_name, 1))
        in_queue, out_queue = queues[i - 1 : i + 1]

        match cfg.type_:
            case _PType.Pipe | _PType.Aggregate | _PType.Disaggregate:
                coro = _pipe(in_queue, out_queue, cfg.args, report_stats_interval)
            case _PType.OrderedPipe:
                coro = _ordered_pipe(
                    in_queue, out_queue, cfg.args, report_stats_interval
                )
            case _:  # pragma: no cover
                raise ValueError(f"Unexpected process type: {cfg.type_}")

        coros.append((f"Pipeline::{cfg.args.name}", coro))

    # sink
    n = len(process_args) + 1
    queue_class = _get_queue(sink.queue_class, report_stats_interval)
    output_queue = queue_class(f"{n}:sink_queue", sink.buffer_size)
    coros.append(
        (
            f"Pipeline::{n}:sink",
            _sink(queues[-1], output_queue),
        )
    )

    return _run_pipeline_coroutines(coros), output_queue


################################################################################
# Coroutine execution logics
################################################################################


# TODO [Python 3.11]: Migrate to ExceptionGroup
class PipelineFailure(RuntimeError):
    """PipelineFailure()
    Thrown by :py:class:`spdl.pipeline.Pipeline` when pipeline encounters an error.
    """

    def __init__(self, errs: dict[str, Exception]) -> None:
        msg = []
        for k, v in errs.items():
            e = str(v)
            msg.append(f"{k}:{e if e else type(v).__name__}")
        msg.sort()

        super().__init__(", ".join(msg))

        # This is for unittesting.
        self._errs = errs


async def _run_pipeline_coroutines(
    coros: list[tuple[str, Coroutine[None, None, None]]],
) -> None:
    """Run the pipeline coroutines and handle errors.

    Args:
        coros: The croutines each corresponds to a stage in pipelin.
            IMPORTANT: The coroutinues must be in the order of src to sink.
    """
    tasks = [create_task(coro, name=name) for name, coro in coros]
    pending = set(tasks)

    while pending:
        # Note:
        # `asyncio.wait` does not automatically propagate the cancellation to its children.
        # For graceful shutdown, we manually cancel the child tasks.
        #
        # Also, it seems asyncio loop throws Cancellation on most outer task.
        # I am not sure where this behavior is documented, but here is an example script to
        # demonstrate the behavior.
        # https://gist.github.com/mthrok/3a1c11c2d8012e29f4835679ac0baaee
        try:
            _, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_EXCEPTION
            )
        except asyncio.CancelledError:
            for task in pending:
                task.cancel()
            await asyncio.wait(pending)
            raise

        if not pending:
            break

        # Check if any of the task caused an error.
        # If an error occured, we cancel the stages upstream to the failed one,
        # then continue waiting the downstream ones.
        for i in range(len(tasks) - 1, -1, -1):
            task = tasks[i]
            if task.done() and not task.cancelled() and task.exception() is not None:
                for task in tasks[:i]:
                    task.cancel()
                break

    errs = {}
    for task in tasks:
        if not task.cancelled() and (err := task.exception()) is not None:
            errs[task.get_name()] = err

    if errs:
        raise PipelineFailure(errs)
