# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from asyncio import Task
from dataclasses import dataclass
from typing import Coroutine, Generic, TypeVar

from ._queue import AsyncQueue
from ._utils import create_task

T = TypeVar("T")


# Used to express the upstream relation ship of coroutines.
# Currently only a chain is used, but we plan to extend it to
# (possibly nested) "Y" shape.
# Retaining relationship is necessary for graceful shutdown.
@dataclass
class _Node(Generic[T]):
    name: str
    coro: Coroutine[None, None, None]
    queue: AsyncQueue[T]
    upstream: list["_Node"]
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
