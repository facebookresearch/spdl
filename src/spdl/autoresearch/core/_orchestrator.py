# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small generic async work scheduler.

Design note
===========

Keep this module domain-neutral. It schedules bounded concurrent coroutine
work, checkpoints queued/running specs, and handles local cancellation. Do not
add SPDL, coding-agent, source-control, metrics, hypothesis-planning, or
experiment-phase logic here. Domain behavior belongs behind the adapter's
coroutines.

If a future change needs source control, builds, job launch/status/progress,
metric collection, or local execution, put that behavior in the workflow or in
the platform capability layer. The runner should stay a small scheduler that
can run any coroutine-producing adapter.

.. mermaid::

   flowchart LR
       Engine["Orchestrator"]
       Adapter["WorkflowProtocol"]
       Domain["Domain coroutine"]
       Checkpoint["adapter.checkpoint"]

       Engine -->|"load()"| Adapter
       Engine -->|"make_coro(TaskSpec)"| Adapter
       Adapter --> Domain
       Domain -->|"TaskResult(children)"| Engine
       Engine -->|"on_result(...)"| Adapter
       Engine --> Checkpoint
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import signal
from collections.abc import Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

_LG: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "Orchestrator",
    "WorkflowProtocol",
    "TaskResult",
    "TaskSpec",
]


@dataclass
class TaskSpec:
    """Serializable unit of work for the async engine."""

    id: str
    """Unique identifier for this work item."""

    priority: float = 0.0
    """Scheduling priority.  Lower values are dequeued first.  The engine
    uses a min-heap, so ``priority=-1000`` runs before ``priority=0``."""

    kind: str = "default"
    """Category tag used by the adapter to distinguish spec types without
    inspecting the payload.  For example, the SPDL workflow sets this to
    ``"experiment"`` so the policy layer can filter experiment specs from
    other internal bookkeeping specs."""

    payload: dict[str, object] = field(default_factory=dict)
    """Arbitrary key-value data that travels with the spec through the
    engine lifecycle (enqueue → checkpoint → make_coro → on_result).  The
    adapter stores domain-specific state here — for example, the SPDL
    workflow stores the full ``HypothesisNode`` dict under
    ``payload["node"]``."""

    def to_dict(self) -> dict[str, object]:
        """Serialize this spec to a JSON-compatible dictionary.

        Returns:
            A dictionary with keys ``"id"``, ``"priority"``, ``"kind"``,
            and ``"payload"``.
        """
        return {
            "id": self.id,
            "priority": self.priority,
            "kind": self.kind,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TaskSpec:
        """Reconstruct a ``TaskSpec`` from a dictionary.

        Missing keys fall back to field defaults.  A non-dict ``payload``
        value is replaced with an empty dictionary.

        Args:
            data: Dictionary previously produced by :py:meth:`to_dict`.

        Returns:
            A new ``TaskSpec`` instance.
        """
        payload = data.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        return cls(
            id=str(data["id"]),
            # pyrefly: ignore [bad-argument-type]
            priority=float(data.get("priority", 0.0)),
            kind=str(data.get("kind", "default")),
            payload=payload,
        )


@dataclass
class TaskResult:
    """Result returned by a completed work coroutine."""

    children: list[TaskSpec] = field(default_factory=list)
    """New work specs to enqueue after this item completes.  The engine
    passes these to ``WorkflowProtocol.on_result`` which may filter or transform
    them before they enter the priority queue."""


class WorkflowProtocol(Protocol):
    """Protocol for domain-specific adapters that drive the work engine.

    Implementations must provide ``load``, ``checkpoint``, ``make_coro``,
    ``on_result``, and ``summarize``.
    """

    def load(self) -> list[TaskSpec]:
        """Return the initial list of work specs to process.

        Called once at the start of ``Orchestrator.run`` when no
        ``initial_specs`` are provided.  Typically reads from a checkpoint
        file to resume interrupted work.
        """
        ...

    def checkpoint(
        self,
        queued: list[TaskSpec],
        running: list[TaskSpec],
        status: str,
    ) -> None:
        """Persist the current scheduler state.

        Called on every engine loop iteration and on interruption.
        ``status`` is one of ``"running"``, ``"stopped"``, or
        ``"interrupted"``.
        """
        ...

    def make_coro(self, spec: TaskSpec) -> Coroutine[Any, Any, TaskResult]:
        """Create the coroutine that executes a work spec.

        The returned coroutine is scheduled as an ``asyncio.Task``.  It
        should return a ``TaskResult`` whose ``children`` field contains
        any follow-up specs to enqueue.
        """
        ...

    async def on_result(self, spec: TaskSpec, result: TaskResult) -> list[TaskSpec]:
        """Process a completed result and return specs to enqueue.

        Called after a work coroutine finishes.  The adapter can filter
        duplicates, update persistent state, or transform the result's
        children before they enter the priority queue.
        """
        ...

    def summarize(self, workdir: Path) -> str:
        """Return a human-readable summary of the workdir state.

        Required. Must be safe to call at any time — before any run, in
        the middle of a run, after a paused or interrupted run, and
        after a clean exit. The framework calls this method to handle
        ``autoresearch summary <wd>`` invocations on demand and writes
        the result to ``<wd>/report.md`` automatically when the engine
        exits cleanly.

        Implementations should render the summary deterministically
        from durable workdir state (master tables, summary files,
        recorded failures), without invoking long-running operations
        such as coding-agent calls.
        """
        ...


class _PriorityQueue:
    def __init__(self, specs: list[TaskSpec] | None = None) -> None:
        self._items: list[tuple[float, int, TaskSpec]] = []
        self._counter = 0
        for spec in specs or []:
            self.push(spec)

    def push(self, spec: TaskSpec) -> None:
        heapq.heappush(self._items, (spec.priority, self._counter, spec))
        self._counter += 1

    def extend(self, specs: list[TaskSpec]) -> None:
        for spec in specs:
            self.push(spec)

    def pop(self) -> TaskSpec | None:
        if not self._items:
            return None
        return heapq.heappop(self._items)[2]

    def items(self) -> list[TaskSpec]:
        return [spec for _, _, spec in sorted(self._items)]

    def __bool__(self) -> bool:
        return bool(self._items)


class Orchestrator:
    """Run serializable work specs with bounded async concurrency.

    Args:
        workflow: Domain-specific adapter implementing the
            :py:class:`WorkflowProtocol` protocol.
        max_concurrency: Maximum number of work coroutines to run
            concurrently (clamped to at least 1).

    .. mermaid::

       sequenceDiagram
           participant Engine as Orchestrator
           participant Adapter as WorkflowProtocol
           participant Task as Work coroutine

           Engine->>Adapter: load()
           Engine->>Adapter: checkpoint(queued, running, "running")
           Engine->>Adapter: make_coro(spec)
           Engine->>Task: schedule up to max_concurrency
           Task-->>Engine: TaskResult(children)
           Engine->>Adapter: on_result(spec, result)
           Adapter-->>Engine: child TaskSpecs
           Engine->>Adapter: checkpoint(..., "stopped")

           Note over Engine,Adapter: On SIGINT/SIGTERM, cancel tasks and
           Note over Engine,Adapter: checkpoint "interrupted".
    """

    def __init__(
        self,
        *,
        workflow: WorkflowProtocol,
        max_concurrency: int,
    ) -> None:
        self.adapter = workflow
        self.max_concurrency = max(1, max_concurrency)

    async def run(self, initial_specs: list[TaskSpec] | None = None) -> None:
        """Run the engine until all work is complete or interrupted.

        The engine dequeues specs from a priority queue, schedules up to
        ``max_concurrency`` coroutines at a time, and checkpoints state on
        every iteration.  On ``SIGINT`` / ``SIGTERM`` it cancels running
        tasks and checkpoints with ``"interrupted"`` status so that a
        subsequent run can resume from where it left off.

        Args:
            initial_specs: Specs to seed the priority queue with.  When
                ``None``, the adapter's ``load()`` method is called instead,
                which typically reads a checkpoint file written by a previous
                interrupted run.
        """
        queued = _PriorityQueue(
            self.adapter.load() if initial_specs is None else initial_specs
        )
        running: dict[asyncio.Task[TaskResult], TaskSpec] = {}
        main_task = asyncio.current_task()
        self._install_signal_handlers(main_task)

        try:
            while queued or running:
                self._fill_slots(queued, running)
                self.adapter.checkpoint(
                    queued.items(),
                    list(running.values()),
                    "running",
                )

                if not running:
                    break

                done, _ = await asyncio.wait(
                    running.keys(),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                await self._process_done(done, running, queued)

            self.adapter.checkpoint([], [], "stopped")
        except asyncio.CancelledError:
            await self._cancel_running(running)
            self.adapter.checkpoint(
                queued.items(),
                list(running.values()),
                "interrupted",
            )
            _LG.info("Async work engine interrupted")

    def _fill_slots(
        self,
        queued: _PriorityQueue,
        running: dict[asyncio.Task[TaskResult], TaskSpec],
    ) -> None:
        while len(running) < self.max_concurrency and queued:
            spec = queued.pop()
            if spec is None:
                break
            task = asyncio.create_task(self.adapter.make_coro(spec), name=spec.id)
            running[task] = spec

    async def _process_done(
        self,
        done: set[asyncio.Task[TaskResult]],
        running: dict[asyncio.Task[TaskResult], TaskSpec],
        queued: _PriorityQueue,
    ) -> None:
        for task in done:
            spec = running.pop(task)
            result = task.result()
            children = await self.adapter.on_result(spec, result)
            queued.extend(children)

    async def _cancel_running(
        self,
        running: dict[asyncio.Task[TaskResult], TaskSpec],
    ) -> None:
        for task in running:
            task.cancel()
        if running:
            await asyncio.gather(*running.keys(), return_exceptions=True)

    def _install_signal_handlers(self, main_task: asyncio.Task[object] | None) -> None:
        if main_task is None:
            return
        task = main_task
        loop = asyncio.get_running_loop()

        def _cancel_main(sig_name: str) -> None:
            _LG.info("%s received; interrupting async work engine", sig_name)
            task.cancel()

        try:
            loop.add_signal_handler(signal.SIGINT, _cancel_main, "SIGINT")
            loop.add_signal_handler(signal.SIGTERM, _cancel_main, "SIGTERM")
        except NotImplementedError:
            pass
