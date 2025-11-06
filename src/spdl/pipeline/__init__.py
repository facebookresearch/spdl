# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Implements :py:class:`~spdl.pipeline.Pipeline`, a generic task execution engine."""

# pyre-strict

from ._build import build_pipeline
from ._builder import PipelineBuilder, run_pipeline_in_subprocess
from ._common._misc import create_task
from ._components import (
    AsyncQueue,
    is_eof,
    PipelineFailure,
    QueuePerfStats,
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
from ._pipeline import Pipeline
from ._profile import profile_pipeline, ProfileHook, ProfileResult

__all__ = [
    "build_pipeline",
    "is_eof",
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
    "StatsQueue",
    "QueuePerfStats",
    "iterate_in_subinterpreter",
    "iterate_in_subprocess",
    "run_pipeline_in_subprocess",
]


try:
    from . import fb  # noqa: F401
except ImportError:
    pass
