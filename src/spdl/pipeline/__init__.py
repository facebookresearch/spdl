# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Implements :py:class:`~spdl.pipeline.Pipeline`, a generic task execution engine."""

# pyre-strict

from ._builder import PipelineBuilder, PipelineFailure, run_pipeline_in_subprocess
from ._hook import PipelineHook, TaskStatsHook
from ._pipeline import Pipeline
from ._queue import AsyncQueue, StatsQueue
from ._utils import cache_iterator, create_task, iterate_in_subprocess

__all__ = [
    "Pipeline",
    "PipelineBuilder",
    "PipelineFailure",
    "cache_iterator",
    "create_task",
    "PipelineHook",
    "TaskStatsHook",
    "AsyncQueue",
    "StatsQueue",
    "iterate_in_subprocess",
    "run_pipeline_in_subprocess",
]
