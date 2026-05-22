# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core types and scheduling engine for autoresearch.

Provides the foundational building blocks for running automated experiment
workflows: domain types for tracking experiments and failures, a
bounded-concurrency async orchestrator, a workflow protocol for domain
adapters, and serializable task types.

Example::

    from spdl.autoresearch.core import (
        Orchestrator, TaskSpec, TaskResult, WorkflowProtocol,
    )

    class MyWorkflow(WorkflowProtocol):
        def load(self) -> list[TaskSpec]:
            return []  # or resume from checkpoint

        def checkpoint(self, queued, running, status) -> None:
            ...  # persist scheduler state for resumption

        def make_coro(self, spec: TaskSpec) -> ...:
            # Called for each dequeued TaskSpec.  Return a coroutine
            # that runs the experiment and produces a TaskResult.
            return self._run_experiment(spec)

        async def _run_experiment(self, spec: TaskSpec) -> TaskResult:
            metrics = await launch_and_wait(spec.payload)
            # Return child TaskSpecs to expand the search tree —
            # the engine enqueues them by priority for later execution.
            return TaskResult(children=[
                TaskSpec(id=f"{spec.id}_variant_a", priority=-1),
                TaskSpec(id=f"{spec.id}_variant_b", priority=0),
            ])

        async def on_result(self, spec, result) -> list[TaskSpec]:
            # Filter or transform children before they enter the queue.
            return result.children

    engine = Orchestrator(workflow=MyWorkflow(), max_concurrency=4)
    await engine.run([TaskSpec(id="exp_001", priority=0)])
"""

from ._orchestrator import Orchestrator, TaskResult, TaskSpec, WorkflowProtocol
from ._types import (
    AnalysisResult,
    AutoresearchError,
    FailureKind,
    FailurePhase,
    FailureRecord,
    HypothesisNode,
    TERMINAL_STATUSES,
)

__all__ = [
    "AnalysisResult",
    "AutoresearchError",
    "FailureKind",
    "FailurePhase",
    "FailureRecord",
    "HypothesisNode",
    "Orchestrator",
    "TaskResult",
    "TaskSpec",
    "TERMINAL_STATUSES",
    "WorkflowProtocol",
]
