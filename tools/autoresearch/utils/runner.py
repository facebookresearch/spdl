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
       Engine["AsyncWorkEngine"]
       Adapter["_WorkAdapter"]
       Domain["Domain coroutine"]
       Checkpoint["adapter.checkpoint"]

       Engine -->|"load()"| Adapter
       Engine -->|"make_coro(_WorkSpec)"| Adapter
       Adapter --> Domain
       Domain -->|"_WorkResult(children)"| Engine
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
from typing import Any, Protocol

_LG: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "AsyncWorkEngine",
    "_WorkAdapter",
    "_WorkResult",
    "_WorkSpec",
]


@dataclass
class _WorkSpec:
    id: str
    priority: float = 0.0
    kind: str = "default"
    payload: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "priority": self.priority,
            "kind": self.kind,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> _WorkSpec:
        payload = data.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        return cls(
            id=str(data["id"]),
            priority=float(data.get("priority", 0.0)),
            kind=str(data.get("kind", "default")),
            payload=payload,
        )


@dataclass
class _WorkResult:
    children: list[_WorkSpec] = field(default_factory=list)


class _WorkAdapter(Protocol):
    def load(self) -> list[_WorkSpec]: ...

    def checkpoint(
        self,
        queued: list[_WorkSpec],
        running: list[_WorkSpec],
        status: str,
    ) -> None: ...

    def make_coro(self, spec: _WorkSpec) -> Coroutine[Any, Any, _WorkResult]: ...

    async def on_result(
        self, spec: _WorkSpec, result: _WorkResult
    ) -> list[_WorkSpec]: ...


class _PriorityQueue:
    def __init__(self, specs: list[_WorkSpec] | None = None) -> None:
        self._items: list[tuple[float, int, _WorkSpec]] = []
        self._counter = 0
        for spec in specs or []:
            self.push(spec)

    def push(self, spec: _WorkSpec) -> None:
        heapq.heappush(self._items, (spec.priority, self._counter, spec))
        self._counter += 1

    def extend(self, specs: list[_WorkSpec]) -> None:
        for spec in specs:
            self.push(spec)

    def pop(self) -> _WorkSpec | None:
        if not self._items:
            return None
        return heapq.heappop(self._items)[2]

    def items(self) -> list[_WorkSpec]:
        return [spec for _, _, spec in sorted(self._items)]

    def __bool__(self) -> bool:
        return bool(self._items)


class AsyncWorkEngine:
    """Run serializable work specs with bounded async concurrency.

    .. mermaid::

       sequenceDiagram
           participant Engine as AsyncWorkEngine
           participant Adapter as _WorkAdapter
           participant Task as Work coroutine

           Engine->>Adapter: load()
           Engine->>Adapter: checkpoint(queued, running, "running")
           Engine->>Adapter: make_coro(spec)
           Engine->>Task: schedule up to max_concurrency
           Task-->>Engine: _WorkResult(children)
           Engine->>Adapter: on_result(spec, result)
           Adapter-->>Engine: child WorkSpecs
           Engine->>Adapter: checkpoint(..., "stopped")

           Note over Engine,Adapter: On SIGINT/SIGTERM, cancel tasks and
           Note over Engine,Adapter: checkpoint "interrupted".
    """

    def __init__(
        self,
        *,
        adapter: _WorkAdapter,
        max_concurrency: int,
    ) -> None:
        self.adapter = adapter
        self.max_concurrency = max(1, max_concurrency)

    async def run(self, initial_specs: list[_WorkSpec] | None = None) -> None:
        queued = _PriorityQueue(
            self.adapter.load() if initial_specs is None else initial_specs
        )
        running: dict[asyncio.Task[_WorkResult], _WorkSpec] = {}
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
        running: dict[asyncio.Task[_WorkResult], _WorkSpec],
    ) -> None:
        while len(running) < self.max_concurrency and queued:
            spec = queued.pop()
            if spec is None:
                break
            task = asyncio.create_task(self.adapter.make_coro(spec), name=spec.id)
            running[task] = spec

    async def _process_done(
        self,
        done: set[asyncio.Task[_WorkResult]],
        running: dict[asyncio.Task[_WorkResult], _WorkSpec],
        queued: _PriorityQueue,
    ) -> None:
        for task in done:
            spec = running.pop(task)
            result = task.result()
            children = await self.adapter.on_result(spec, result)
            queued.extend(children)

    async def _cancel_running(
        self,
        running: dict[asyncio.Task[_WorkResult], _WorkSpec],
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
