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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TypeVar

from ._components._pipe import _FailCounter, _get_fail_counter, _ordered_pipe, _pipe
from ._components._sink import _sink
from ._components._source import _source
from ._hook import TaskHook, TaskStatsHook as DefaultHook
from ._node import _cancel_upstreams_of_errors, _gather_error, _Node, _start_tasks
from ._pipeline import Pipeline
from ._queue import AsyncQueue, StatsQueue as DefaultQueue
from .defs._defs import _PipeType, PipeConfig, PipelineConfig, SinkConfig, SourceConfig

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
#    (It is sink's responsibility to filter out EOF.)
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
#      - The failed stage must pass EOF to the next queue, so that
#        the downstream stages can exit cleanly.
#      - Passing EOF to the next queue can be blocked if the queue is full.
#      - For a graceful exit, the assumption of the continued consumption must be met.
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


def _get_names(base: str, stage_id: int) -> tuple[str, str]:
    # Note: Do not change the pattern used for naming.
    # They are used by dashboard to query runtime data.
    node_name = f"{_PIPELINE_ID}:{stage_id}:{base}"
    queue_name = f"{node_name}_queue"
    return node_name, queue_name


def _build_src_node(
    cfg: SourceConfig[T], stage_id: int, q_class: type[AsyncQueue[T]]
) -> _Node[T]:
    node_name, q_name = _get_names("src", stage_id)
    q = q_class(q_name, buffer_size=2)
    return _Node(name=node_name, coro=_source(cfg.source, q), queue=q, upstream=[])


def _build_sink_node(
    cfg: SinkConfig[T],
    stage_id: int,
    q_class: type[AsyncQueue[T]],
    prev: _Node[T],
) -> _Node[T]:
    node_name, q_name = _get_names("sink", stage_id)
    in_q: AsyncQueue[T] = prev.queue
    out_q = q_class(q_name, buffer_size=cfg.buffer_size)
    return _Node(name=node_name, coro=_sink(in_q, out_q), queue=out_q, upstream=[prev])


def _build_pipe_node(
    cfg: PipeConfig[T, U],
    stage_id: int,
    q_class: type[AsyncQueue[U]],
    prev: _Node[T],
    task_hook_factory: Callable[[str], list[TaskHook]],
    fc: _FailCounter,
) -> _Node[U]:
    pipe_name = cfg.name
    if cfg._type == _PipeType.Pipe and cfg._args.concurrency > 1:
        pipe_name = f"{pipe_name}[{cfg._args.concurrency}]"

    node_name, q_name = _get_names(pipe_name, stage_id)
    hooks = task_hook_factory(node_name)
    # Use buffer_size=2 so that it is possible that queue always
    # has an item as long as upstream is fast enough.
    # This make it possible for data readiness (occupancy rate)
    # to reach 100%, instead of 99.999999%
    in_q, out_q = prev.queue, q_class(q_name, buffer_size=2)

    match cfg._type:
        case _PipeType.Pipe | _PipeType.Aggregate | _PipeType.Disaggregate:
            coro = _pipe(pipe_name, in_q, out_q, cfg._args, fc, hooks)
        case _PipeType.OrderedPipe:
            coro = _ordered_pipe(pipe_name, in_q, out_q, cfg._args, fc, hooks)
        case _:  # pragma: no cover
            raise ValueError(f"Unexpected process type: {cfg._type}")

    return _Node(name=node_name, coro=coro, queue=out_q, upstream=[prev])


def _build_pipeline_node(
    plc: PipelineConfig[T, U],
    stage_id: int,
    q_class: type[AsyncQueue[...]],
    fc_class: type[_FailCounter],
    task_hook_factory: Callable[[str], list[TaskHook]],
    max_failures: int,
) -> _Node[U]:
    global _PIPELINE_ID
    _PIPELINE_ID += 1

    node = _build_src_node(plc.src, stage_id, q_class)  # pyre-ignore[6]
    stage_id += 1

    for cfg in plc.pipes:
        fc = fc_class(max_failures, cfg._max_failures)
        prev: _Node[T] = node  # pyre-ignore[9]
        node = _build_pipe_node(cfg, stage_id, q_class, prev, task_hook_factory, fc)
        stage_id += 1

    node = _build_sink_node(plc.sink, stage_id, q_class, node)  # pyre-ignore[6]
    return node


################################################################################
# Coroutine execution logics
################################################################################


# TODO [Python 3.11]: Migrate to ExceptionGroup
class PipelineFailure(RuntimeError):
    """PipelineFailure()
    Thrown by :py:class:`spdl.pipeline.Pipeline` when pipeline encounters an error.
    """

    def __init__(self, errs: list[tuple[str, Exception]]) -> None:
        msg = []
        for k, v in errs:
            e = str(v)
            msg.append(f"{k}:{e if e else type(v).__name__}")
        msg.sort()

        super().__init__(", ".join(msg))

        # This is for unittesting.
        self._errs = errs


async def _run_pipeline_coroutines(node: _Node[T]) -> None:
    """Run the pipeline coroutines and handle errors."""
    pending = _start_tasks(node)

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
        if canceled := _cancel_upstreams_of_errors(node):
            await asyncio.wait(canceled)

    if errs := _gather_error(node):
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

    _fail_counter_class = _get_fail_counter()

    node = _build_pipeline_node(
        pipeline_cfg,
        stage_id,
        _queue_class,
        _fail_counter_class,
        _hook_factory if task_hook_factory is None else task_hook_factory,
        max_failures,
    )
    coro = _run_pipeline_coroutines(node)

    executor = ThreadPoolExecutor(
        max_workers=num_threads,
        thread_name_prefix="spdl_worker_thread_",
    )
    return Pipeline(coro, node.queue, executor, desc=desc)


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
