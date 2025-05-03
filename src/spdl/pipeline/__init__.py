# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Implements :py:class:`~spdl.pipeline.Pipeline`, a generic task execution engine."""

# pyre-strict

from ._builder import PipelineBuilder, PipelineFailure, run_pipeline_in_subprocess
from ._hook import TaskHook, TaskPerfStats, TaskStatsHook
from ._pipeline import Pipeline
from ._queue import AsyncQueue, QueuePerfStats, StatsQueue
from ._utils import cache_iterator, create_task, iterate_in_subprocess

__all__ = [
    "Pipeline",
    "PipelineBuilder",
    "PipelineFailure",
    "cache_iterator",
    "create_task",
    "TaskHook",
    "TaskStatsHook",
    "TaskPerfStats",
    "AsyncQueue",
    "StatsQueue",
    "QueuePerfStats",
    "iterate_in_subprocess",
    "run_pipeline_in_subprocess",
]


try:
    from . import fb  # noqa: F401
except ImportError:
    pass


def __getattr__(name: str) -> type[TaskHook]:
    if name == "PipelineHook":
        import warnings

        warnings.warn(
            "spdl.pipeline.PipelineHook has been renamed to spdl.pipeline.TaskHook. "
            "Please change the import path.",
            stacklevel=2,
        )

        return TaskHook
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
