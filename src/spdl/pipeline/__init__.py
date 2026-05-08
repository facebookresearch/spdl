# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Implements :py:class:`~spdl.pipeline.Pipeline`, a generic task execution engine."""

# pyre-strict

from ._bg_task import BackgroundTask, BackgroundTaskFactory
from ._build import (
    build_pipeline,
    run_pipeline_in_subinterpreter,
    run_pipeline_in_subprocess,
)
from ._builder import PipelineBuilder
from ._common._misc import create_task
from ._components import (
    AsyncQueue,
    is_eof,
    is_epoch_end,
    PipelineFailure,
    QueuePerfStats,
    StageInfo,
    StatsQueue,
    TaskHook,
    TaskPerfStats,
    TaskStatsHook,
)
from ._iter_utils import (
    cache_iterator,
    iterate_in_subinterpreter,
    iterate_in_subprocess,
)
from ._pgrp_stats import (
    ProcessGroupResourceUsage,
    ProcessGroupStatsMonitor,
)
from ._pipeline import Pipeline
from ._priority_executor import (
    PriorityExecutorEntrypoint,
    PriorityProcessPoolExecutor,
    PriorityThreadPoolExecutor,
)
from ._priority_interpreter_executor import PriorityInterpreterPoolExecutor
from ._profile import profile_pipeline, ProfileHook, ProfileResult

__all__ = [
    "BackgroundTask",
    "BackgroundTaskFactory",
    "ProcessGroupResourceUsage",
    "ProcessGroupStatsMonitor",
    "build_pipeline",
    "is_eof",
    "is_epoch_end",
    "profile_pipeline",
    "ProfileResult",
    "ProfileHook",
    "Pipeline",
    "PipelineBuilder",
    "PipelineFailure",
    "cache_iterator",
    "create_task",
    "TaskHook",
    "TaskStatsHook",
    "TaskPerfStats",
    "AsyncQueue",
    "StageInfo",
    "StatsQueue",
    "QueuePerfStats",
    "PriorityExecutorEntrypoint",
    "PriorityInterpreterPoolExecutor",
    "PriorityThreadPoolExecutor",
    "PriorityProcessPoolExecutor",
    "iterate_in_subinterpreter",
    "iterate_in_subprocess",
    "run_pipeline_in_subprocess",
    "run_pipeline_in_subinterpreter",
]


try:
    from . import fb  # noqa: F401
except ImportError:
    pass
