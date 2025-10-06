# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from asyncio import Task
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Coroutine, Generic, TypeVar

from ._queue import AsyncQueue
from ._utils import create_task

T = TypeVar("T")

__all__ = [
    "_Node",
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
    coro: Coroutine[None, None, None]
    queue: AsyncQueue[T]
    upstream: "Sequence[_Node]"
    _task: Task | None = None

    @property
    def task(self) -> Task:
        if self._task is None:
            raise AssertionError("[INTERNAL ERROR] task is not started.")
        return self._task

    def create_task(self) -> Task:
        if self._task is not None:
            raise AssertionError("[INTERNAL ERROR] task is already started.")
        task = create_task(self.coro, name=self.name)
        self._task = task
        return task


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
