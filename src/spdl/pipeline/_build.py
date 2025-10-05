# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "_build_pipeline",
    "build_pipeline",
    "PipelineFailure",
]

import asyncio
import logging
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TypeVar

from ._components._pipe import _get_fail_counter, _ordered_pipe, _pipe
from ._components._sink import _sink
from ._components._source import _source
from ._hook import TaskHook, TaskStatsHook as DefaultHook
from ._pipeline import Pipeline
from ._queue import AsyncQueue, StatsQueue as DefaultQueue
from ._utils import create_task
from .defs._defs import _PipeType, PipeConfig, PipelineConfig

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


# Used to append stage name with pipeline
_PIPELINE_ID: int = -1


def _get_pipe_prefix(cfg: PipeConfig[..., ...]) -> str:
    name = cfg.name
    if cfg._type == _PipeType.Pipe and cfg._args.concurrency > 1:
        name = f"{name}[{cfg._args.concurrency}]"
    return name


def _build_pipeline_coro(
    plc: PipelineConfig[T, U],
    max_failures: int,
    queue_class: type[AsyncQueue[...]],
    task_hook_factory: Callable[[str], list[TaskHook]],
    stage_id: int,
) -> tuple[Coroutine[None, None, None], AsyncQueue[U]]:
    # Note:
    # Make sure that coroutines are ordered from source to sink.
    # `_run_pipeline_coroutines` expects and rely on this ordering.
    coros = []

    global _PIPELINE_ID
    _PIPELINE_ID += 1

    # source
    coro_name = f"{stage_id}:src"
    q_name = f"{_PIPELINE_ID}:{coro_name}_queue"
    stage_id += 1
    out_q = queue_class(q_name)
    coros.append((coro_name, _source(plc.src.source, out_q)))

    _FailCounter = _get_fail_counter()

    # pipes
    for cfg in plc.pipes:
        pipe_name = _get_pipe_prefix(cfg)
        coro_name = f"{stage_id}:{pipe_name}"
        q_name = f"{_PIPELINE_ID}:{coro_name}_queue"
        stage_id += 1
        # Use buffer_size=2 so that it is possible that queue always
        # has an item as long as upstream is fast enough.
        # This make it possible for data readiness (occupancy rate)
        # to reach 100%, instead of 99.999999%
        in_q, out_q = out_q, queue_class(q_name, buffer_size=2)
        fc = _FailCounter(max_failures, cfg._max_failures)
        hook = task_hook_factory(f"{_PIPELINE_ID}:{coro_name}")

        match cfg._type:
            case _PipeType.Pipe | _PipeType.Aggregate | _PipeType.Disaggregate:
                coro = _pipe(pipe_name, in_q, out_q, cfg._args, fc, hook)
            case _PipeType.OrderedPipe:
                coro = _ordered_pipe(pipe_name, in_q, out_q, cfg._args, fc, hook)
            case _:  # pragma: no cover
                raise ValueError(f"Unexpected process type: {cfg._type}")

        coros.append((coro_name, coro))

    # sink
    coro_name = f"{stage_id}:sink"
    q_name = f"{_PIPELINE_ID}:{coro_name}_queue"
    in_q, out_q = out_q, queue_class(q_name, buffer_size=plc.sink.buffer_size)
    coros.append((coro_name, _sink(in_q, out_q)))

    return _run_pipeline_coroutines(coros), out_q  # pyre-ignore


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
                if upstream := tasks[:i]:
                    for task in upstream:
                        task.cancel()
                    await asyncio.wait(upstream)
                break

    errs = {}
    for task in tasks:
        if not task.cancelled() and (err := task.exception()) is not None:
            errs[task.get_name()] = err

    if errs:
        raise PipelineFailure(errs)


def _build_pipeline(
    pipeline_cfg: PipelineConfig[T, U],
    /,
    *,
    num_threads: int,
    max_failures: int = -1,
    report_stats_interval: float = -1,
    queue_class: type[AsyncQueue[...]] | None = None,
    task_hook_factory: Callable[[str], list[TaskHook]] | None = None,
    stage_id: int = 0,
) -> Pipeline[U]:
    desc = repr(pipeline_cfg)

    _LG.debug("%s", desc)

    def _hook_factory(name: str) -> list[TaskHook]:
        return [DefaultHook(name=name, interval=report_stats_interval)]

    _queue_class = (
        partial(DefaultQueue, interval=report_stats_interval)
        if queue_class is None
        else queue_class
    )

    coro, queues = _build_pipeline_coro(
        pipeline_cfg,
        max_failures,
        _queue_class,
        _hook_factory if task_hook_factory is None else task_hook_factory,
        stage_id,
    )

    executor = ThreadPoolExecutor(
        max_workers=num_threads,
        thread_name_prefix="spdl_worker_thread_",
    )
    return Pipeline(coro, queues, executor, desc=desc)


def build_pipeline(
    pipeline_cfg: PipelineConfig[T, U],
    /,
    *,
    num_threads: int,
    max_failures: int = -1,
    report_stats_interval: float = -1,
    queue_class: type[AsyncQueue[...]] | None = None,
    task_hook_factory: Callable[[str], list[TaskHook]] | None = None,
    stage_id: int = 0,
) -> Pipeline[U]:
    """Build a pipeline from the config.

    Args:
        pipeline_cfg: The definition of the pipeline to build.

        num_threads: The number of threads in the thread pool commonly used among
            Pipeline stages.

            .. note::

               If a stage in the pipeline has dedicated executor, that stage will
               use it.

        max_failures: The maximum number of failures each pipe stage can have before
            the pipeline is halted. Setting ``-1`` (default) disables it.

        report_stats_interval: When provided, report the pipeline performance stats
            every given interval. Unit: [sec]

            This is only effective if there is no custom hook or custom AsyncQueue
            provided for stages. The argument is passed to
            :py:class:`TaskStatsHook` and :py:class:`StatsQueue`.

            If a custom stage hook is provided and stats report is needed,
            you can instantiate :py:class:`TaskStatsHook` and include
            it in the hooks provided to :py:meth:`PipelineBuilder.pipe`.

            Similarly if you are providing a custom :py:class:`AsyncQueue` class,
            you need to implement the same logic by your self.

        queue_class: If provided, override the queue class used to connect stages.
            Must be a class (not an instance) inherits :py:class:`AsyncQueue`.

        task_hook_factory: If provided, used to create task hook objects, given a
            name of the stage. If ``None``, a default hook,
            :py:class:`TaskStatsHook` is used.
            To disable hooks, provide a function that returns an empty list.

        stage_id: The index of the initial stage  used for logging.
    """
    return _build_pipeline(
        pipeline_cfg,
        num_threads=num_threads,
        max_failures=max_failures,
        report_stats_interval=report_stats_interval,
        queue_class=queue_class,
        task_hook_factory=task_hook_factory,
        stage_id=stage_id,
    )
