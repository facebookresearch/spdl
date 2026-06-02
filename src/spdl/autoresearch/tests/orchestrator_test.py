# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from spdl.autoresearch.core import Orchestrator, TaskResult, TaskSpec

__all__: list[str] = []


class _FakeAdapter:
    def __init__(self, specs: list[TaskSpec]) -> None:
        self.specs = specs
        self.started: list[str] = []
        self.checkpoints: list[tuple[list[str], list[str], str]] = []
        self.children: dict[str, list[TaskSpec]] = {}
        self.block = False

    def load(self) -> list[TaskSpec]:
        return self.specs

    def checkpoint(
        self,
        queued: list[TaskSpec],
        running: list[TaskSpec],
        status: str,
    ) -> None:
        self.checkpoints.append(
            ([spec.id for spec in queued], [spec.id for spec in running], status)
        )

    async def make_coro(self, spec: TaskSpec) -> TaskResult:
        self.started.append(spec.id)
        if self.block:
            await asyncio.sleep(60)
        return TaskResult(children=self.children.get(spec.id, []))

    async def on_result(self, spec: TaskSpec, result: TaskResult) -> list[TaskSpec]:
        return result.children

    def summarize(self, workdir: Path) -> str:
        return f"_FakeAdapter summary at {workdir}"


class OrchestratorTest(unittest.IsolatedAsyncioTestCase):
    async def test_priority_order_lowest_first(self) -> None:
        """Specs are executed in ascending priority order (lowest value first)."""
        adapter = _FakeAdapter(
            [
                TaskSpec(id="slow", priority=10),
                TaskSpec(id="first", priority=-1),
                TaskSpec(id="middle", priority=5),
            ]
        )

        await Orchestrator(workflow=adapter, max_concurrency=1).run()

        self.assertEqual(["first", "middle", "slow"], adapter.started)
        self.assertEqual(([], [], "stopped"), adapter.checkpoints[-1])

    async def test_completion_enqueues_children(self) -> None:
        """Child specs returned by a completed item are enqueued and executed."""
        adapter = _FakeAdapter([TaskSpec(id="root", priority=0)])
        adapter.children["root"] = [
            TaskSpec(id="child_a", priority=1),
            TaskSpec(id="child_b", priority=2),
        ]

        await Orchestrator(workflow=adapter, max_concurrency=1).run()

        self.assertEqual(["root", "child_a", "child_b"], adapter.started)
        self.assertEqual(([], [], "stopped"), adapter.checkpoints[-1])

    async def test_cancelled_error_persists_interrupted_state(self) -> None:
        """Cancellation checkpoints running specs with 'interrupted' status."""
        adapter = _FakeAdapter([TaskSpec(id="running", priority=0)])
        adapter.block = True
        task = asyncio.create_task(
            Orchestrator(workflow=adapter, max_concurrency=1).run()
        )

        while not adapter.checkpoints:
            await asyncio.sleep(0)
        task.cancel()
        await task

        self.assertEqual(([], ["running"], "interrupted"), adapter.checkpoints[-1])

    async def test_checkpoint_resume_golden_lifecycle(self) -> None:
        """An interrupted engine can resume from checkpointed state and complete."""
        first = _FakeAdapter(
            [
                TaskSpec(id="running", priority=0),
                TaskSpec(id="queued", priority=1),
            ]
        )
        first.block = True
        task = asyncio.create_task(
            Orchestrator(workflow=first, max_concurrency=1).run()
        )

        while not first.checkpoints:
            await asyncio.sleep(0)
        task.cancel()
        await task

        queued_ids, running_ids, status = first.checkpoints[-1]
        self.assertEqual(
            (["queued"], ["running"], "interrupted"), first.checkpoints[-1]
        )

        resumed_specs = [
            TaskSpec(id=spec_id, priority=0 if spec_id == "running" else 1)
            for spec_id in running_ids + queued_ids
        ]
        resumed = _FakeAdapter(resumed_specs)
        await Orchestrator(workflow=resumed, max_concurrency=1).run()

        self.assertEqual("interrupted", status)
        self.assertEqual(["running", "queued"], resumed.started)
        self.assertEqual(([], [], "stopped"), resumed.checkpoints[-1])
