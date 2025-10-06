# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from asyncio import Task
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Coroutine, Generic, TypeVar

from ._components._pipe import _FailCounter, _ordered_pipe, _pipe
from ._components._sink import _sink
from ._components._source import _source
from ._hook import TaskHook
from ._queue import AsyncQueue
from ._utils import create_task
from .defs._defs import (
    _ConfigBase,
    _PipeType,
    PipeConfig,
    PipelineConfig,
    SinkConfig,
    SourceConfig,
)

T = TypeVar("T")

__all__ = [
    "_Node",
    "_build_pipeline_node",
    "_run_pipeline_coroutines",
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
    queue: AsyncQueue[T]
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


def _get_names(cfg: _ConfigBase, pipeline_id: int, stage_id: int) -> tuple[str, str]:
    # Note: Do not change the pattern used for naming.
    # They are used by dashboard to query runtime data.
    match cfg:
        case SourceConfig():
            base = "src"
        case SinkConfig():
            base = "sink"
        case PipeConfig():
            base = cfg.name
            if cfg._type == _PipeType.Pipe and cfg._args.concurrency > 1:
                base = f"{base}[{cfg._args.concurrency}]"
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
    q_class: type[AsyncQueue[...]],
    pipeline_id: int,
    stage_id: int,
) -> _Node:
    name, q_name = _get_names(plc.src, pipeline_id, stage_id)
    stage_id += 1
    n = _Node(name, plc.src, [], q_class(q_name, buffer_size=_BUFFER_SIZE))
    for cfg in plc.pipes:
        name, q_name = _get_names(cfg, pipeline_id, stage_id)
        stage_id += 1
        n = _Node(name, cfg, [n], q_class(q_name, buffer_size=_BUFFER_SIZE))
    name, q_name = _get_names(plc.sink, pipeline_id, stage_id)
    stage_id += 1
    n = _Node(name, plc.sink, [n], q_class(q_name, buffer_size=plc.sink.buffer_size))
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
                case _PipeType.Pipe | _PipeType.Aggregate | _PipeType.Disaggregate:
                    node._coro = _pipe(node.name, in_q, out_q, cfg._args, fc, hooks)
                case _PipeType.OrderedPipe:
                    node._coro = _ordered_pipe(
                        node.name, in_q, out_q, cfg._args, fc, hooks
                    )
                case _:  # pragma: no cover
                    raise ValueError(f"Unexpected process type: {cfg._type}")


def _build_pipeline_node(
    plc: PipelineConfig,
    pipeline_id: int,
    stage_id: int,
    q_class: type[AsyncQueue[...]],
    fc_class: type[_FailCounter],
    task_hook_factory: Callable[[str], list[TaskHook]],
    max_failures: int,
):
    node = _convert_config(plc, q_class, pipeline_id, stage_id)
    _build_node(node, fc_class, task_hook_factory, max_failures)
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
