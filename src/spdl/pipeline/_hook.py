# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from asyncio import Task
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator, Sequence
from contextlib import asynccontextmanager, AsyncExitStack, contextmanager
from dataclasses import dataclass
from typing import AsyncContextManager, TypeVar

from ._utils import create_task

__all__ = [
    "_stage_hooks",
    "_task_hooks",
    "_time_str",
    "_StatsCounter",
    "TaskHook",
    "TaskStatsHook",
    "TaskPerfStats",
]

_LG: logging.Logger = logging.getLogger(__name__)


T = TypeVar("T")


def _time_str(val: float) -> str:
    if val < 0.0001:
        val *= 1e6
        unit = "us"
    elif val < 1:
        val *= 1e3
        unit = "ms"
    else:
        unit = "sec"

    return f"{val:6.1f} [{unit:>3s}]"


class _StatsCounter:
    def __init__(self) -> None:
        self._n: int = 0
        self._t: float = 0.0

    @property
    def num_items(self) -> int:
        return self._n

    @property
    def ave_time(self) -> float:
        return self._t

    def update(self, t: float, n: int = 1) -> None:
        if n > 0:
            self._n += n
            self._t += (t - self._t) * n / self._n

    @contextmanager
    def count(self) -> Iterator[None]:
        t0 = time.monotonic()
        yield
        elapsed = time.monotonic() - t0
        self.update(elapsed)


class TaskHook(ABC):
    """Base class for hooks to be used in the pipeline.

    ``TaskHook`` can add custom actions when executing pipeline.
    It is useful for logging and profiling the pipeline execution.

    A hook consists of two async context managers. ``stage_hook`` and ``task_hook``.

    ``stage_hook`` is executed once when the pipeline is initialized and finalized.
    ``task_hook`` is executed for each task.

    The following diagram illustrates this.

    .. mermaid::

       flowchart TD
           subgraph TaskGroup[Tasks ]
               subgraph Task1[Task 1]
                   direction TB
                   s1["__aenter__() from task_hook()"] --> task1["task()"]
                   task1 --> e1["__aexit__() from task_hook()"]
                   e1 -.-> |"If task() succeeded, \\nand __aexit__() did not fail"| q1["Result Queued"]
               end
               subgraph ...
                   direction TB
                   foo[...]
               end
               subgraph TaskN[Task N]
                   direction TB
                   sN["__aenter__() from task_hook()"] --> taskN["task()"]
                   taskN --> eN["__aexit__() from task_hook()"]
                   eN -.-> |"If task() succeeded, \\nand __aexit__() did not fail"| qN["Result Queued"]
               end
           end
           StageStart["__aenter__() from stage_hook()"] --> TaskGroup
           TaskGroup --> StageComplete["__aexit__() from stage_hook()"]

    To add custom hook, subclass this class and override ``task_hook`` and
    optionally ``stage_hook`` method, and pass an instance to methods such as
    :py:meth:`spdl.pipeline.Pipeline.pipe`.

    .. tip::

       When implementing a hook, you can decide how to react to task/stage failure, by
       choosing the location of specific logics.

       See :py:obj:`contextlib.contextmanager` for detail, and
       :py:class:`spdl.pipeline.TaskStatsHook` for an example implementation.

       .. code-block:: python

          @asynccontextmanager
          async def stage_hook(self):
              # Add initialization logic here
              ...

              try:
                  yield
                  # Add logic that should be executed only when stage succeeds
                  ...
              except Exception as e:
                  # Add logic to react to specific exceptions
                  ...
              finally:
                  # Add logic that should be executed even if stage fails
                  ...

    .. important::

        When implementing a task hook, make sure that ``StopAsyncIteration``
        exception is not absorbed. Otherwise, if `pipe` is given an async generator
        the pipeline might run forever.

        .. code-block:: python

          @asynccontextmanager
          async def stage_hook(self):
              # Add initialization logic here
              ...

              try:
                  yield
                  # Add logic that should be executed only when stage succeeds
                  ...
              except StopAsyncIteration:
                  # When passing async generator to `pipe`, StopAsyncIteration is raised
                  # from inside and will be caught here.
                  # Do no absort it and propagate it to the other.
                  # Usually, you do not want to do anything here.
                  raise
              except Exception as e:
                  # Add logic to react to specific exceptions
                  ...
              finally:
                  # Add logic that should be executed even if stage fails
                  ...

    Args:
        name: The name of the stage. Assigned by :py:class:`~spdl.pipeline.PipelineBuilder`.

    """

    @asynccontextmanager
    async def stage_hook(self) -> AsyncIterator[None]:
        """Perform custom action when the pipeline stage is initialized and completed.

        .. important::

           This method has to be async context manager. So when overriding the method,
           make sure to use ``async`` keyword and ``@asynccontextmanager`` decorator.

           .. code-block:: python

              @asynccontextmanager
              async def stage_hook(self):
                  # Add custom logic here

        .. caution::

           If this hook raises an exception, the pipeline is aborted.
        """
        yield

    @abstractmethod
    @asynccontextmanager
    async def task_hook(self) -> AsyncIterator[None]:
        """Perform custom action before and after task is executed.

        .. important::

           This method has to be async context manager. So when overriding the method,
           make sure to use ``async`` keyword and ``@asynccontextmanager`` decorator.

           .. code-block:: python

              @asynccontextmanager
              async def stask_hook(self):
                  # Add custom logic here

        .. note::

           This method is called as part of the task.
           Even if this method raises an exception, the pipeline is not aborted.
           However, the data being processed is dropped.
        """
        yield


def _stage_hooks(hooks: Sequence[TaskHook]) -> AsyncContextManager[None]:
    hs: list[AsyncContextManager[None]] = [hook.stage_hook() for hook in hooks]

    if not all(hasattr(h, "__aenter__") and hasattr(h, "__aexit__") for h in hs):
        raise ValueError(
            "`stage_hook()` must return an object that has `__aenter__` and"
            " `__aexit__` method. "
            "Make sure that `stage_hook()` is decorated with `asynccontextmanager`."
        )

    @asynccontextmanager
    async def stage_hooks() -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            for h in hs:
                await stack.enter_async_context(h)
            yield

    return stage_hooks()


def _task_hooks(hooks: Sequence[TaskHook]) -> AsyncContextManager[None]:
    hs: list[AsyncContextManager[None]] = [hook.task_hook() for hook in hooks]

    if not all(hasattr(h, "__aenter__") or hasattr(h, "__aexit__") for h in hs):
        raise ValueError(
            "`task_hook()` must return an object that has `__aenter__` and"
            " `__aexit__` method. "
            "Make sure that `task_hook()` is decorated with `asynccontextmanager`."
        )

    @asynccontextmanager
    async def task_hooks() -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            for h in hs:
                await stack.enter_async_context(h)
            yield

    return task_hooks()


async def _periodic_dispatch(
    afun: Callable[[], Coroutine[None, None, None]],
    done: asyncio.Event,
    interval: float,
) -> None:
    assert interval > 0, "[InternalError] `interval` must be greater than 0."
    pending: set[Task] = set()
    target = time.monotonic() + interval
    while True:
        if (dt := target - time.monotonic()) > 0:
            await asyncio.wait([create_task(done.wait())], timeout=dt)

        if done.is_set():
            break

        target = time.monotonic() + interval
        pending.add(create_task(afun()))

        # Assumption interval >> task duration.
        _, pending = await asyncio.wait(pending, return_when="FIRST_COMPLETED")

    if pending:
        await asyncio.wait(pending)


@dataclass
class TaskPerfStats:
    """Performance statistics of a task measured by :py:class:`TaskStatsHook`.

    .. seealso::

       - :ref:`logging` explains how to export the runtime performance statistics.
       - :ref:`analysis` explains how to use the exported stats.
    """

    num_tasks: int
    """The number of tasks invoked."""

    num_failures: int
    """The number of tasks failed."""

    ave_time: float
    """The average execution time (in second) of successful tasks."""


class TaskStatsHook(TaskHook):
    """Track the task runtimes and success rate.

    .. seealso::

       - :ref:`logging` explains how to export the runtime performance statistics.
       - :ref:`analysis` explains how to use the exported stats.

    Args:
        name: Name of the stage. Only used for logging.
        interval: The interval (in second) to report the performance stats periodically.
            The default behavior is no periodic reporting.
    """

    def __init__(
        self,
        name: str,
        interval: float = -1,
    ) -> None:
        assert interval is not None
        self.name = name
        self.interval = interval

        self.num_tasks = 0
        self.num_success = 0
        self.ave_time = 0.0

        # For interval
        self._lap_t0 = 0.0
        self._lap_num_tasks = 0
        self._lap_num_success = 0
        self._lap_ave_time = 0.0

    @asynccontextmanager
    async def stage_hook(self) -> AsyncIterator[None]:
        """Track the stage runtime and log the task stats."""
        if self.interval > 0:
            self._lap_t0 = time.monotonic()
            done = asyncio.Event()
            report = create_task(
                _periodic_dispatch(self._log_interval_stats, done, self.interval),
                name="f{self.name}_periodic_report",
                log_cancelled=False,
            )

        try:
            yield
        finally:
            if self.interval > 0:
                done.set()
                await report  # pyre-ignore: [61]
            self._log_stats(
                TaskPerfStats(
                    num_tasks=self.num_tasks,
                    num_failures=self.num_tasks - self.num_success,
                    ave_time=self.ave_time,
                ),
            )

    @asynccontextmanager
    async def task_hook(self) -> AsyncIterator[None]:
        """Track task runtime and success rate."""
        t0 = time.monotonic()
        try:
            yield
        except StopAsyncIteration:
            raise
        except Exception:
            self.num_tasks += 1
            raise
        else:
            # We only track the average runtime of successful tasks.
            elapsed = time.monotonic() - t0
            self.num_tasks += 1
            self.num_success += 1
            self.ave_time += (elapsed - self.ave_time) / self.num_success

    def _get_lap_stats(self) -> TaskPerfStats:
        num_success = self.num_success
        num_tasks = self.num_tasks
        ave_time = self.ave_time

        delta_num_tasks = max(0, num_tasks - self._lap_num_tasks)
        delta_num_success = max(0, num_success - self._lap_num_success)
        if delta_num_success <= 0:
            delta_ave_time = 0.0
        else:
            total_time = ave_time * num_success
            lap_total_time = self._lap_ave_time * self._lap_num_success
            delta_ave_time = max(0.0, (total_time - lap_total_time) / delta_num_success)

        self._lap_num_tasks = num_tasks
        self._lap_num_success = num_success
        self._lap_ave_time = ave_time

        return TaskPerfStats(
            num_tasks=delta_num_tasks,
            num_failures=delta_num_tasks - delta_num_success,
            ave_time=delta_ave_time,
        )

    async def _log_interval_stats(self) -> None:
        await self.interval_stats_callback(self._get_lap_stats())

    async def interval_stats_callback(self, stats: TaskPerfStats) -> None:
        """Callback for processing interval performance statistics.

        When interval reporting is enabled, this method is periodically called
        with the delta metrics.

        The default behavior is to log the metrics to console.

        You can override this method and modify the destination of the log.
        """
        self._log_stats(stats)

    def _log_stats(self, stats: TaskPerfStats) -> None:
        _LG.info(
            "[%s]\tCompleted %5d tasks (%3d failed). Ave task time: %s.",
            self.name,
            stats.num_tasks,
            stats.num_failures,
            _time_str(stats.ave_time),
            stacklevel=2,
        )
