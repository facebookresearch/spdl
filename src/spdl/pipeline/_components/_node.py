# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from asyncio import Task
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, TypeVar

from spdl.pipeline._common._misc import create_task
from spdl.pipeline.defs import (
    _ConfigBase,
    _PipeArgs,
    _PipeType,
    AggregateConfig,
    DisaggregateConfig,
    MergeConfig,
    PipeConfig,
    PipelineConfig,
    SinkConfig,
    SourceConfig,
)

from ._hook import get_default_hook_class, TaskHook
from ._pipe import (
    _Aggregate,
    _disaggregate,
    _FailCounter,
    _get_fail_counter,
    _merge,
    _ordered_pipe,
    _pipe,
)
from ._queue import AsyncQueue, get_default_queue_class
from ._sink import _sink
from ._source import _source

T = TypeVar("T")

__all__ = [
    "_Node",
    "_build_pipeline_coro",
    "PipelineFailure",
]


# Used to express the upstream relation ship of coroutines.
# Currently only a chain is used, but we plan to extend it to
# (possibly nested) "Y" shape.
# Retaining relationship is necessary for graceful shutdown.
@dataclass
class _Node(Generic[T]):
    name: str
    cfg: _ConfigBase
    upstream: "Sequence[_Node]"
    queue: AsyncQueue
    _coro: Coroutine[None, None, None] | None = None
    _task: Task | None = None

    @property
    def coro(self) -> Coroutine[None, None, None]:
        if self._coro is None:
            raise RuntimeError(
                "[Internal Error] Attempted to retrieve a coroutine object, "
                f"but it is not created: {self}"
            )
        return self._coro

    @property
    def task(self) -> Task:
        if self._task is None:
            raise AssertionError(
                "[INTERNAL ERROR] Attempted to retrieve a task object, "
                f"but it is not created: {self}"
            )
        return self._task

    def create_task(self) -> Task:
        if self._task is not None:
            raise AssertionError(
                "[INTERNAL ERROR] Attempted to cteate a task, "
                f"but it is already crated (started): {self}"
            )
        task = create_task(self.coro, name=self.name)
        self._task = task
        return task


################################################################################
# Build logic
################################################################################


class _MutableInt:
    def __init__(self, v: int) -> None:
        self.v = v

    def __iadd__(self, v: int) -> "_MutableInt":
        self.v += v
        return self

    def __repr__(self) -> str:
        return str(self.v)


def _get_names(
    cfg: _ConfigBase, pipeline_id: int, stage_id: _MutableInt
) -> tuple[str, str]:
    # Note: Do not change the pattern used for naming.
    # They are used by dashboard to query runtime data.
    match cfg:
        case SourceConfig():
            base = "src"
        case SinkConfig():
            base = "sink"
        case PipeConfig():
            base = cfg.name
            if cfg._args.concurrency > 1:
                base = f"{base}[{cfg._args.concurrency}]"
        case AggregateConfig():
            base = f"aggregate({cfg.num_items}, drop_last={cfg.drop_last})"
        case DisaggregateConfig():
            base = "disaggregate"
        case MergeConfig():
            base = "merge"
        case _:
            raise NotImplementedError(f"`{type(cfg)}` is not supported.")
    name = f"{pipeline_id}:{stage_id}:{base}"
    return name, f"{name}_queue"


# For queues other than Sink, we use buffer_size=2.
# This makes it possible for queues to always have an item
# as long as upstream is fast enough.
# This make it possible for data readiness (occupancy rate)
# to reach 100%, instead of 99.999999%
_BUFFER_SIZE: int = 2


def _convert_config(
    plc: PipelineConfig,
    q_class: type[AsyncQueue],
    pipeline_id: int,
    stage_id: _MutableInt,
    disable_sink: bool = False,
) -> _Node:
    name, q_name = _get_names(plc.src, pipeline_id, stage_id)
    stage_id += 1

    upstream = (
        [
            _convert_config(cfg, q_class, pipeline_id, stage_id, disable_sink=True)
            for cfg in plc.src.pipeline_configs
        ]
        if isinstance(plc.src, MergeConfig)
        else []
    )
    n = _Node(name, plc.src, upstream, q_class(q_name, buffer_size=_BUFFER_SIZE))

    for cfg in plc.pipes:
        name, q_name = _get_names(cfg, pipeline_id, stage_id)
        stage_id += 1
        n = _Node(name, cfg, [n], q_class(q_name, buffer_size=_BUFFER_SIZE))

    if not disable_sink:
        name, q_name = _get_names(plc.sink, pipeline_id, stage_id)
        stage_id += 1
        n = _Node(
            name, plc.sink, [n], q_class(q_name, buffer_size=plc.sink.buffer_size)
        )
    return n


def _build_node(
    node: _Node,
    fc_class: type[_FailCounter],
    task_hook_factory: Callable[[str], list[TaskHook]],
    max_failures: int,
) -> None:
    if node._coro is not None:
        raise RuntimeError(f"[Internal Error] coroutine cannot be built twice. {node}")

    for n in node.upstream:
        _build_node(n, fc_class, task_hook_factory, max_failures)

    match node.cfg:
        case SourceConfig():
            cfg: SourceConfig = node.cfg
            assert len(node.upstream) == 0
            node._coro = _source(cfg.source, node.queue)
        case MergeConfig():
            cfg: MergeConfig = node.cfg
            assert len(node.upstream) > 0
            input_queues = [n.queue for n in node.upstream]
            hooks = task_hook_factory(node.name)
            fc = fc_class(max_failures)
            node._coro = _merge(node.name, input_queues, node.queue, fc, hooks)
        case SinkConfig():
            cfg: SinkConfig = node.cfg
            assert len(node.upstream) == 1
            node._coro = _sink(node.upstream[0].queue, node.queue)
        case PipeConfig():
            cfg: PipeConfig = node.cfg
            assert len(node.upstream) == 1

            in_q, out_q = node.upstream[0].queue, node.queue
            hooks = task_hook_factory(node.name)
            fc = fc_class(max_failures, cfg._max_failures)
            match cfg._type:
                case _PipeType.Pipe:
                    node._coro = _pipe(node.name, in_q, out_q, cfg._args, fc, hooks)
                case _PipeType.OrderedPipe:
                    node._coro = _ordered_pipe(
                        node.name, in_q, out_q, cfg._args, fc, hooks
                    )
                case _:  # pragma: no cover
                    raise ValueError(f"Unexpected process type: {cfg._type}")

        case AggregateConfig():
            cfg: AggregateConfig = node.cfg
            assert len(node.upstream) == 1

            in_q, out_q = node.upstream[0].queue, node.queue
            hooks = task_hook_factory(node.name)
            fc = fc_class(max_failures)
            args = _PipeArgs(
                op=_Aggregate(cfg.num_items, cfg.drop_last),
                op_requires_eof=True,
            )
            node._coro = _pipe(node.name, in_q, out_q, args, fc, hooks)

        case DisaggregateConfig():
            cfg: DisaggregateConfig = node.cfg
            assert len(node.upstream) == 1

            in_q, out_q = node.upstream[0].queue, node.queue
            hooks = task_hook_factory(node.name)
            fc = fc_class(max_failures)
            args = _PipeArgs(op=_disaggregate)  # pyre-ignore[6]
            node._coro = _pipe(node.name, in_q, out_q, args, fc, hooks)


# Used to append stage name with pipeline
_PIPELINE_ID: int = -1


def _default_q(interval: float) -> type[AsyncQueue]:
    queue_class = get_default_queue_class()
    return partial(queue_class, interval=interval)  # pyre-ignore[7]


def _default_hook_factory(
    report_stats_interval: float,
) -> Callable[[str], list[TaskHook]]:
    if (hook_class := get_default_hook_class()) is not None:

        def _hook_factory(name: str) -> list[TaskHook]:
            return [hook_class(name=name, interval=report_stats_interval)]

    else:

        def _hook_factory(_: str) -> list[TaskHook]:
            return []

    return _hook_factory


def _build_pipeline_node(
    plc: PipelineConfig,
    /,
    *,
    max_failures: int,
    report_stats_interval: float,
    queue_class: type[AsyncQueue] | None,
    task_hook_factory: Callable[[str], list[TaskHook]] | None,
    stage_id: int,
):
    global _PIPELINE_ID
    _PIPELINE_ID += 1

    q_class = _default_q(report_stats_interval) if queue_class is None else queue_class
    hook_factory = (
        _default_hook_factory(report_stats_interval)
        if task_hook_factory is None
        else task_hook_factory
    )

    fc_class = _get_fail_counter()
    node = _convert_config(plc, q_class, _PIPELINE_ID, _MutableInt(stage_id))
    _build_node(node, fc_class, hook_factory, max_failures)
    return node


################################################################################
# Coroutine execution logics
################################################################################
def _start_tasks(node: _Node[T]) -> set[Task]:
    node.create_task()
    ret = {node.task}
    for n in node.upstream:
        ret |= _start_tasks(n)
    return ret


def _cancel_recursive(node: _Node[T]) -> set[Task]:
    task = node.task
    task.cancel()
    ret = {task}
    for n in node.upstream:
        ret |= _cancel_recursive(n)
    return ret


def _cancel_upstreams_of_errors(node: _Node[T]) -> set[Task]:
    task = node.task
    canceled = set()
    if task.done() and not task.cancelled() and task.exception() is not None:
        for n in node.upstream:
            canceled |= _cancel_recursive(n)

    for n in node.upstream:
        canceled |= _cancel_upstreams_of_errors(n)
    return canceled


def _gather_error(node: _Node[T]) -> list[tuple[str, Exception]]:
    task = node.task
    errs = []
    if not task.cancelled() and (err := task.exception()) is not None:
        errs.append((task.get_name(), err))

    for n in node.upstream:
        errs.extend(_gather_error(n))
    errs.sort(key=lambda i: i[0])
    return errs


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


def _build_pipeline_coro(
    plc: PipelineConfig[Any],
    /,
    *,
    max_failures: int = -1,
    report_stats_interval: float = -1,
    queue_class: type[AsyncQueue] | None = None,
    task_hook_factory: Callable[[str], list[TaskHook]] | None = None,
    stage_id: int = 0,
) -> tuple[Coroutine[None, None, None], asyncio.Queue]:
    node = _build_pipeline_node(
        plc,
        max_failures=max_failures,
        report_stats_interval=report_stats_interval,
        queue_class=queue_class,
        task_hook_factory=task_hook_factory,
        stage_id=stage_id,
    )
    coro = _run_pipeline_coroutines(node)

    return coro, node.queue
