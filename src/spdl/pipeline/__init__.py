# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Implements :py:class:`~spdl.pipeline.Pipeline`, a generic task execution engine."""

# pyre-strict

from ._builder import PipelineBuilder, PipelineFailure
from ._hook import PipelineHook, TaskStatsHook
from ._pipeline import Pipeline
from ._utils import create_task, iterate_in_subprocess

__all__ = [
    "Pipeline",
    "PipelineBuilder",
    "PipelineFailure",
    "create_task",
    "PipelineHook",
    "TaskStatsHook",
    "iterate_in_subprocess",
]
