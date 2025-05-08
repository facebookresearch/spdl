# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "_build_pipeline_coro",
    "PipelineFailure",
    "_SourceConfig",
    "_ProcessConfig",
    "_SinkConfig",
]

import asyncio
import enum
import inspect
from collections.abc import AsyncIterable, Callable, Coroutine, Iterable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .._hook import TaskHook
from .._queue import AsyncQueue
from .._utils import create_task
from ._pipe import _FailCounter, _ordered_pipe, _pipe, _PipeArgs
from ._sink import _sink
from ._source import _source

# pyre-strict

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class _SourceConfig(Generic[T]):
    source: Iterable | AsyncIterable

    def __post_init__(self) -> None:
        if not (hasattr(self.source, "__aiter__") or hasattr(self.source, "__iter__")):
            raise ValueError("Source must be either generator or async generator.")


class _PType(enum.IntEnum):
    Pipe = 1
    OrderedPipe = 2
    Aggregate = 3
    Disaggregate = 4


@dataclass
class _ProcessConfig(Generic[T, U]):
    type_: _PType
    name: str
    args: _PipeArgs[T, U]

    def __post_init__(self) -> None:
        op = self.args.op
        if inspect.iscoroutinefunction(op) or inspect.isasyncgenfunction(op):
            if self.args.executor is not None:
                raise ValueError("`executor` cannot be specified when op is async.")
        if inspect.isasyncgenfunction(op):
            if self.type_ == _PType.OrderedPipe:
                raise ValueError(
                    "pipe does not support async generator function "
                    "when `output_order` is 'input'."
                )


@dataclass
class _SinkConfig(Generic[T]):
    buffer_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.buffer_size, int):
            raise ValueError(
                f"`buffer_size` must be int. Found: {type(self.buffer_size)}."
            )
        if self.buffer_size < 1:
            raise ValueError(
                f"`buffer_size` must be greater than 0. Found: {self.buffer_size}"
            )


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


def _get_task_name(i: int, cfg: _ProcessConfig[..., ...]) -> str:
    name = f"{_PIPELINE_ID}:{i}:{cfg.name}"
    if cfg.type_ == _PType.Pipe and cfg.args.concurrency > 1:
        name = f"{name}[{cfg.args.concurrency}]"
    return name


def _build_pipeline_coro(
    src: _SourceConfig[T],
    process_args: list[_ProcessConfig[Any, Any]],  # pyre-ignore: [2]
    sink: _SinkConfig[U],
    max_failures: int,
    queue_class: type[AsyncQueue[...]],
    task_hook_factory: Callable[[str], list[TaskHook]],
) -> tuple[Coroutine[None, None, None], AsyncQueue[U]]:
    # Note:
    # Make sure that coroutines are ordered from source to sink.
    # `_run_pipeline_coroutines` expects and rely on this ordering.
    coros = []
    queues = []

    global _PIPELINE_ID
    _PIPELINE_ID += 1

    # source
    queues.append(queue_class(f"{_PIPELINE_ID}:0:src_queue"))
    coros.append(("Pipeline::0:src", _source(src.source, queues[0])))

    _FailCounter = _get_fail_counter()

    # pipes
    for i, cfg in enumerate(process_args, start=1):
        name = _get_task_name(i, cfg)
        queue_name = f"{name}_queue"
        # Use buffer_size=2 so that it is possible that queue always
        # has an item as long as upstream is fast enough.
        # This make it possible for data readiness (occupancy rate)
        # to reach 100%, instead of 99.999999%
        queues.append(queue_class(queue_name, buffer_size=2))
        in_queue, out_queue = queues[i - 1 : i + 1]

        match cfg.type_:
            case _PType.Pipe | _PType.Aggregate | _PType.Disaggregate:
                coro = _pipe(
                    name,
                    in_queue,
                    out_queue,
                    cfg.args,
                    _FailCounter(),
                    task_hook_factory(name),
                    max_failures,
                )
            case _PType.OrderedPipe:
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
    n = len(process_args) + 1
    output_queue = queue_class(
        f"{_PIPELINE_ID}:{n}:sink_queue", buffer_size=sink.buffer_size
    )
    coros.append(
        (
            f"Pipeline::{n}:sink",
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
