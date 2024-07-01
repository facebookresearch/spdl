# pyre-unsafe

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import asynccontextmanager, AsyncExitStack
from typing import TypeVar

__all__ = [
    "PipelineHook",
    "TaskStatsHook",
]

_LG = logging.getLogger(__name__)


T = TypeVar("T")


class PipelineHook(ABC):
    """Base class for hooks to be used in the pipeline.

    A hook consists of two async context managers. ``stage_hook`` and ``task_hook``.

    ``stage_hook`` is executed once when the pipeline stage is initialized and finalized.
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

    .. tip::

       When implementing a hook, you can decide how to react to task/stage failure, by
       choosing the location of specific logics.

       See :py:obj:`contextlib.contextmanager` for detail, and
       :py:class:`spdl.dataloader.TaskStatsHook` for an example implementation.

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


    """

    @asynccontextmanager
    async def stage_hook(self):
        """Perform custom action when the pipeline stage is initialized and completed.

        .. caution::

           If this hook raises an exception, the pipeline is aborted.
        """
        yield

    @abstractmethod
    @asynccontextmanager
    async def task_hook(self):
        """Perform custom action before and after task is executed.

        .. note::

           This method is called as part of the task.
           Even if this method raises an exception, the pipeline is not aborted.
           However, the data being processed is dropped.
        """
        yield


@asynccontextmanager
async def _stage_hooks(hooks: Sequence[PipelineHook]):
    async with AsyncExitStack() as stack:
        for h in hooks:
            stage_hook = h.stage_hook()
            if not hasattr(stage_hook, "__aenter__") or not hasattr(
                stage_hook, "__aexit__"
            ):
                raise ValueError(
                    "`stage_hook()` must return an object that has `__aenter__` and"
                    " `__aexit__` method. "
                    "Make sure that `stage_hook()` is decorated with `asynccontextmanager`."
                )
            await stack.enter_async_context(stage_hook)
        yield


@asynccontextmanager
async def _task_hooks(hooks: Sequence[PipelineHook]):
    async with AsyncExitStack() as stack:
        for h in hooks:
            task_hook = h.task_hook()
            if not hasattr(task_hook, "__aenter__") or not hasattr(
                task_hook, "__aexit__"
            ):
                raise ValueError(
                    "`task_hook()` must return an object that has `__aenter__` and"
                    " `__aexit__` method. "
                    "Make sure that `task_hook()` is decorated with `asynccontextmanager`."
                )
            await stack.enter_async_context(task_hook)
        yield


class TaskStatsHook(PipelineHook):
    """Track the task runtimes and success rate.

    Args:
        name: Nmae of the stage. Only used for logging.
        concurrency: Concurrency of the stage. Only used for logging.
    """

    def __init__(self, name: str, concurrency: int):
        self.name = name
        self.concurrency = concurrency

        self.num_tasks = 0
        self.num_success = 0
        self.ave_time = 0.0

    @asynccontextmanager
    async def stage_hook(self):
        """Track the stage runtime and log the task stats."""
        t0 = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - t0
            self._log_stats(elapsed, self.num_tasks, self.num_success, self.ave_time)

    @asynccontextmanager
    async def task_hook(self):
        """Track task runtime and success rate."""
        self.num_tasks += 1
        t0 = time.monotonic()
        yield
        # We only track the average runtime of successful tasks.
        elapsed = time.monotonic() - t0
        self.num_success += 1
        self.ave_time += (elapsed - self.ave_time) / self.num_success

    def _log_stats(self, elapsed, num_tasks, num_success, ave_time):
        _LG.info(
            "[%s]\tCompleted %5d tasks (%3d failed) in %.4f [%3s]. "
            "QPS: %.2f (Concurrency: %3d). "
            "Average task time: %.4f [%3s].",
            self.name,
            num_tasks,
            num_tasks - num_success,
            elapsed * 1000 if elapsed < 1 else elapsed,
            "ms" if elapsed < 1 else "sec",
            num_success / elapsed if elapsed > 0.001 else float("nan"),
            self.concurrency,
            ave_time * 1000 if ave_time < 1 else ave_time,
            "ms" if ave_time < 1 else "sec",
            stacklevel=2,
        )
