# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import unittest

from spdl.tools.autoresearch.utils.runner import (
    _WorkResult,
    _WorkSpec,
    AsyncWorkEngine,
)

__all__: list[str] = []


class _FakeAdapter:
    def __init__(self, specs: list[_WorkSpec]) -> None:
        self.specs = specs
        self.started: list[str] = []
        self.checkpoints: list[tuple[list[str], list[str], str]] = []
        self.children: dict[str, list[_WorkSpec]] = {}
        self.block = False

    def load(self) -> list[_WorkSpec]:
        return self.specs

    def checkpoint(
        self,
        queued: list[_WorkSpec],
        running: list[_WorkSpec],
        status: str,
    ) -> None:
        self.checkpoints.append(
            ([spec.id for spec in queued], [spec.id for spec in running], status)
        )

    async def make_coro(self, spec: _WorkSpec) -> _WorkResult:
        self.started.append(spec.id)
        if self.block:
            await asyncio.sleep(60)
        return _WorkResult(children=self.children.get(spec.id, []))

    async def on_result(self, spec: _WorkSpec, result: _WorkResult) -> list[_WorkSpec]:
        return result.children


class _SimpleEngineTest(unittest.IsolatedAsyncioTestCase):
    async def test_priority_order_lowest_first(self) -> None:
        adapter = _FakeAdapter(
            [
                _WorkSpec(id="slow", priority=10),
                _WorkSpec(id="first", priority=-1),
                _WorkSpec(id="middle", priority=5),
            ]
        )

        await AsyncWorkEngine(adapter=adapter, max_concurrency=1).run()

        self.assertEqual(["first", "middle", "slow"], adapter.started)
        self.assertEqual(([], [], "stopped"), adapter.checkpoints[-1])

    async def test_completion_enqueues_children(self) -> None:
        adapter = _FakeAdapter([_WorkSpec(id="root", priority=0)])
        adapter.children["root"] = [
            _WorkSpec(id="child_a", priority=1),
            _WorkSpec(id="child_b", priority=2),
        ]

        await AsyncWorkEngine(adapter=adapter, max_concurrency=1).run()

        self.assertEqual(["root", "child_a", "child_b"], adapter.started)
        self.assertEqual(([], [], "stopped"), adapter.checkpoints[-1])

    async def test_cancelled_error_persists_interrupted_state(self) -> None:
        adapter = _FakeAdapter([_WorkSpec(id="running", priority=0)])
        adapter.block = True
        task = asyncio.create_task(
            AsyncWorkEngine(adapter=adapter, max_concurrency=1).run()
        )

        while not adapter.checkpoints:
            await asyncio.sleep(0)
        task.cancel()
        await task

        self.assertEqual(([], ["running"], "interrupted"), adapter.checkpoints[-1])

    async def test_checkpoint_resume_golden_lifecycle(self) -> None:
        first = _FakeAdapter(
            [
                _WorkSpec(id="running", priority=0),
                _WorkSpec(id="queued", priority=1),
            ]
        )
        first.block = True
        task = asyncio.create_task(
            AsyncWorkEngine(adapter=first, max_concurrency=1).run()
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
            _WorkSpec(id=spec_id, priority=0 if spec_id == "running" else 1)
            for spec_id in running_ids + queued_ids
        ]
        resumed = _FakeAdapter(resumed_specs)
        await AsyncWorkEngine(adapter=resumed, max_concurrency=1).run()

        self.assertEqual("interrupted", status)
        self.assertEqual(["running", "queued"], resumed.started)
        self.assertEqual(([], [], "stopped"), resumed.checkpoints[-1])
