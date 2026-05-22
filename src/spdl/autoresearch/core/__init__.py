# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core scheduling engine for autoresearch.

Provides the domain-neutral building blocks for running automated experiment
workflows: a bounded-concurrency async orchestrator, a workflow protocol for
domain adapters, and serializable task types.

Example::

    from spdl.autoresearch.core import Orchestrator, TaskSpec

    engine = Orchestrator(workflow=my_adapter, max_concurrency=4)
    await engine.run([TaskSpec(id="exp_001", priority=0)])
"""

from ._orchestrator import Orchestrator, TaskResult, TaskSpec, WorkflowProtocol

__all__ = [
    "Orchestrator",
    "TaskResult",
    "TaskSpec",
    "WorkflowProtocol",
]
