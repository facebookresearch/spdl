# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "_build_pipeline",
    "_get_desc",
    "PipelineFailure",
]

import asyncio
import logging
import pprint
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, TypeVar

from ._components._pipe import _FailCounter, _ordered_pipe, _pipe
from ._components._sink import _sink
from ._components._source import _source
from ._defs import _PipeConfig, _PipeType, _SinkConfig, _SourceConfig
from ._hook import TaskHook, TaskStatsHook as DefaultHook
from ._pipeline import Pipeline
from ._queue import AsyncQueue, StatsQueue as DefaultQueue
from ._utils import create_task

# pyre-strict

T = TypeVar("T")
U = TypeVar("U")

_LG: logging.Logger = logging.getLogger(__name__)


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


# Use different class variables per pipeline
def _get_fail_counter() -> type[_FailCounter]:
    class _FC(_FailCounter):
        num_failures: int = 0

    return _FC


# Used to append stage name with pipeline
_PIPELINE_ID: int = -1


def _get_task_name(i: int, cfg: _PipeConfig[..., ...]) -> str:
    name = f"{_PIPELINE_ID}:{i}:{cfg.name}"
    if cfg.type_ == _PipeType.Pipe and cfg.args.concurrency > 1:
        name = f"{name}[{cfg.args.concurrency}]"
    return name


def _build_pipeline_coro(
    src: _SourceConfig[T],
    process_args: list[_PipeConfig[Any, Any]],  # pyre-ignore: [2]
    sink: _SinkConfig[U],
    max_failures: int,
    queue_class: type[AsyncQueue[...]],
    task_hook_factory: Callable[[str], list[TaskHook]],
    stage_id: int,
) -> tuple[Coroutine[None, None, None], AsyncQueue[U]]:
    if _LG.isEnabledFor(logging.DEBUG):
        _LG.debug(
            pprint.pformat(
                {
                    "src": src,
                    "pipe": process_args,
                    "sink": sink,
                },
                indent=2,
                sort_dicts=False,
                compact=True,
            ),
        )

    # Note:
    # Make sure that coroutines are ordered from source to sink.
    # `_run_pipeline_coroutines` expects and rely on this ordering.
    coros = []
    queues = []

    global _PIPELINE_ID
    _PIPELINE_ID += 1

    # source
    name = f"{stage_id}:src"
    stage_id += 1
    queues.append(queue_class(f"{_PIPELINE_ID}:{name}_queue"))
    coros.append((f"Pipeline::{name}", _source(src.source, queues[0])))

    _FailCounter = _get_fail_counter()

    # pipes
    for cfg in process_args:
        name = _get_task_name(stage_id, cfg)
        stage_id += 1
        queue_name = f"{name}_queue"
        # Use buffer_size=2 so that it is possible that queue always
        # has an item as long as upstream is fast enough.
        # This make it possible for data readiness (occupancy rate)
        # to reach 100%, instead of 99.999999%
        queues.append(queue_class(queue_name, buffer_size=2))
        in_queue, out_queue = queues[-2:]

        match cfg.type_:
            case _PipeType.Pipe | _PipeType.Aggregate | _PipeType.Disaggregate:
                coro = _pipe(
                    name,
                    in_queue,
                    out_queue,
                    cfg.args,
                    _FailCounter(),
                    task_hook_factory(name),
                    max_failures,
                )
            case _PipeType.OrderedPipe:
                coro = _ordered_pipe(
                    name,
                    in_queue,
                    out_queue,
                    cfg.args,
                    _FailCounter(),
                    task_hook_factory(name),
                    max_failures,
                )
            case _:  # pragma: no cover
                raise ValueError(f"Unexpected process type: {cfg.type_}")

        coros.append((f"Pipeline::{name}", coro))

    # sink
    name = f"{stage_id}:sink_queue"
    output_queue = queue_class(f"{_PIPELINE_ID}:{name}", buffer_size=sink.buffer_size)
    coros.append(
        (
            f"Pipeline::{name}",
            _sink(queues[-1], output_queue),
        )
    )

    return _run_pipeline_coroutines(coros), output_queue  # pyre-ignore


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
        coros: The coroutines each corresponds to a stage in pipeline.
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
        # If an error occurred, we cancel the stages upstream to the failed one,
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


def _get_desc(
    src: _SourceConfig[T] | None,
    process_args: list[_PipeConfig],  # pyre-ignore: [24]
    sink: _SinkConfig[U] | None,
) -> str:
    parts = []
    if (src_ := src) is not None:
        src_repr = getattr(src_.source, "__name__", type(src_.source).__name__)
        parts.append(f"  - src: {src_repr}")
    else:
        parts.append("  - src: n/a")

    for cfg in process_args:
        args = cfg.args
        match cfg.type_:
            case _PipeType.Pipe:
                part = f"{cfg.name}(concurrency={args.concurrency})"
            case _PipeType.OrderedPipe:
                part = (
                    f"{cfg.name}(concurrency={args.concurrency}, "
                    'output_order="input")'
                )
            case _PipeType.Aggregate | _PipeType.Disaggregate:
                part = cfg.name
            case _:
                part = str(cfg.type_)
        parts.append(f"  - {part}")

    if (sink_ := sink) is not None:
        parts.append(f"  - sink: buffer_size={sink_.buffer_size}")

    return "\n".join(parts)


def _build_pipeline(
    src: _SourceConfig[T],
    process_args: list[_PipeConfig],  # pyre-ignore: [24]
    sink: _SinkConfig[U],
    *,
    num_threads: int,
    max_failures: int,
    report_stats_interval: float = -1,
    queue_class: type[AsyncQueue[...]] | None,
    task_hook_factory: Callable[[str], list[TaskHook]] | None,
    stage_id: int,
) -> Pipeline[U]:
    def _hook_factory(name: str) -> list[TaskHook]:
        return [DefaultHook(name=name, interval=report_stats_interval)]

    _queue_class = (
        partial(DefaultQueue, interval=report_stats_interval)
        if queue_class is None
        else queue_class
    )

    coro, queues = _build_pipeline_coro(
        src,
        process_args,
        sink,
        max_failures,
        _queue_class,
        _hook_factory if task_hook_factory is None else task_hook_factory,
        stage_id,
    )

    executor = ThreadPoolExecutor(
        max_workers=num_threads,
        thread_name_prefix="spdl_worker_thread_",
    )
    return Pipeline(coro, queues, executor, desc=_get_desc(src, process_args, sink))
