# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Implements :py:class:`~spdl.pipeline.Pipeline`, a generic task execution engine."""

from ._arena import SharedMemoryRingBuffer, SharedMemorySegmentPool
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
    "AsyncQueue",
    "BackgroundTask",
    "BackgroundTaskFactory",
    "Pipeline",
    "PipelineBuilder",
    "PipelineFailure",
    "PriorityExecutorEntrypoint",
    "PriorityInterpreterPoolExecutor",
    "PriorityProcessPoolExecutor",
    "PriorityThreadPoolExecutor",
    "ProcessGroupResourceUsage",
    "ProcessGroupStatsMonitor",
    "ProfileHook",
    "ProfileResult",
    "QueuePerfStats",
    "SharedMemoryRingBuffer",
    "SharedMemorySegmentPool",
    "StageInfo",
    "StatsQueue",
    "TaskHook",
    "TaskPerfStats",
    "TaskStatsHook",
    "build_pipeline",
    "cache_iterator",
    "create_task",
    "is_eof",
    "is_epoch_end",
    "iterate_in_subinterpreter",
    "iterate_in_subprocess",
    "profile_pipeline",
    "run_pipeline_in_subinterpreter",
    "run_pipeline_in_subprocess",
]


try:
    from . import fb  # noqa: F401
except ImportError:
    pass
