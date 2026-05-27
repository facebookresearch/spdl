# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core types and scheduling engine for autoresearch.

Provides the foundational building blocks for running automated experiment
workflows: domain types for tracking experiments and failures, a
bounded-concurrency async orchestrator, a workflow protocol for domain
adapters, serializable task types, and default JSON-backed engine-state
persistence helpers.

Example::

    from pathlib import Path

    from spdl.autoresearch.core import (
        Orchestrator, TaskResult, TaskSpec, WorkflowProtocol,
        load_or_init, write_engine_state,
    )

    class MyWorkflow(WorkflowProtocol):
        def __init__(self, workdir: Path) -> None:
            self.workdir = workdir

        def load(self) -> list[TaskSpec]:
            return load_or_init(self.workdir, self._initial_specs)

        def checkpoint(self, queued, running, status) -> None:
            write_engine_state(
                self.workdir,
                queued=queued,
                running=running,
                status=status,
            )

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

        def summarize(self, workdir: Path) -> str:
            # Render workdir state as markdown. Safe to call any time.
            return _render_summary(workdir)

        def _initial_specs(self) -> list[TaskSpec]:
            return [TaskSpec(id="exp_001", priority=0)]

    engine = Orchestrator(workflow=MyWorkflow(Path("/tmp/run")), max_concurrency=4)
    await engine.run()
"""

from ._orchestrator import Orchestrator, TaskResult, TaskSpec, WorkflowProtocol
from ._persistence import load_or_init, read_engine_state, write_engine_state
from ._types import (
    AnalysisResult,
    AutoresearchError,
    FailureKind,
    FailurePhase,
    FailureRecord,
    HypothesisNode,
    TERMINAL_STATUSES,
)
from ._workflow import WorkflowFactory, WorkflowSpec

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
    "WorkflowFactory",
    "WorkflowProtocol",
    "WorkflowSpec",
    "load_or_init",
    "read_engine_state",
    "write_engine_state",
]
