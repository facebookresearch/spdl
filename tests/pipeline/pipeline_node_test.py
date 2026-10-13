# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import asyncio
import unittest

from spdl.pipeline._components import AsyncQueue
from spdl.pipeline._components._common import StageInfo
from spdl.pipeline._components._node import (
    _cancel_orphaned,
    _cancel_recursive,
    _FanInNode,
    _gather_error,
    _Node,
    _PathVariantsMergeConfig,
    _SourceNode,
    _start_tasks,
)
from spdl.pipeline.defs import SinkConfig, SourceConfig


class DummyException(Exception):
    pass


_TTestNode = _SourceNode | _Node | _FanInNode


def _node(
    name: str,
    deps: list[_TTestNode],
    exc: Exception | None = None,
    gate: asyncio.Event | None = None,
) -> _TTestNode:
    async def coro() -> None:
        if exc:
            raise exc
        elif gate is not None:
            await gate.wait()
        else:
            await asyncio.sleep(10)

    info = StageInfo(pipeline_id=0, stage_id="0", stage_name=name)
    n: _TTestNode
    if not deps:
        n = _SourceNode(
            info,
            SourceConfig(source=[]),
            output_queue=AsyncQueue(info),
        )
    elif len(deps) > 1:
        n = _FanInNode(
            info,
            _PathVariantsMergeConfig(),
            deps,
            input_queues=[
                AsyncQueue(
                    StageInfo(pipeline_id=0, stage_id="0", stage_name=f"{name}_in_{i}")
                )
                for i in range(len(deps))
            ],
            output_queue=AsyncQueue(info),
        )
    else:
        n = _Node(
            info,
            SinkConfig(buffer_size=1),
            deps,
            input_queue=AsyncQueue(
                StageInfo(pipeline_id=0, stage_id="0", stage_name=f"{name}_in")
            ),
            output_queue=AsyncQueue(info),
        )
    n._coro = coro()
    return n


class PipelineNodeAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_completed_branch_keeps_shared_producer_alive(self) -> None:
        """A shared producer is cancelled only after every consumer finishes."""
        left_done, right_done = asyncio.Event(), asyncio.Event()
        source = _node("source", [])
        shared = _node("shared", [source])
        exclusive = _node("exclusive", [shared])
        left = _node("left", [exclusive], gate=left_done)
        right = _node("right", [shared], gate=right_done)
        sink = _node("sink", [left, right])
        tasks = [
            n.create_task() for n in (source, shared, exclusive, left, right, sink)
        ]
        try:
            left_done.set()
            await left.task
            _cancel_orphaned(sink)
            await asyncio.sleep(0)
            self.assertTrue(exclusive.task.cancelled())
            self.assertFalse(shared.task.done())
            self.assertFalse(source.task.done())
            self.assertFalse(right.task.done())

            right_done.set()
            await right.task
            _cancel_orphaned(sink)
            await asyncio.wait([shared.task, source.task])
            self.assertTrue(shared.task.cancelled())
            self.assertTrue(source.task.cancelled())
        finally:
            _cancel_recursive(sink)
            await asyncio.gather(*tasks, return_exceptions=True)

    async def test_completed_sink_cancels_exclusive_producer_chain(self) -> None:
        """An early sink exit cancels its entire unused chain in one traversal."""
        done = asyncio.Event()
        source = _node("source", [])
        middle = _node("middle", [source])
        sink = _node("sink", [middle], gate=done)
        tasks = [n.create_task() for n in (source, middle, sink)]
        try:
            done.set()
            await sink.task
            _cancel_orphaned(sink)
            await asyncio.wait([middle.task, source.task])
            self.assertTrue(middle.task.cancelled())
            self.assertTrue(source.task.cancelled())
        finally:
            _cancel_recursive(sink)
            await asyncio.gather(*tasks, return_exceptions=True)


class PipelineNodeTest(unittest.TestCase):
    def test_node_chain_start_and_cancel(self) -> None:
        #  A -> B -> C

        async def run() -> None:
            a = _node("A", [])
            b = _node("B", [a])
            c = _node("C", [b])
            tasks = _start_tasks(c)
            self.assertTrue(all(isinstance(t, asyncio.Task) for t in tasks))
            self.assertEqual(len(tasks), 3)

            _cancel_recursive(c)
            await asyncio.wait(tasks)
            self.assertTrue(all(t.cancelled() for t in tasks))

        asyncio.run(run())

    def test_node_y_shape_upstream(self) -> None:
        #  A1      B1
        #   |      |
        #  A2      B2
        #    \    /
        #      C1

        async def run() -> None:
            a1 = _node("A1", [])
            a2 = _node("A2", [a1])
            b1 = _node("B1", [])
            b2 = _node("B2", [b1])
            c1 = _node("C1", [a2, b2])
            tasks = _start_tasks(c1)
            self.assertEqual(len(tasks), 5)
            _cancel_recursive(c1)
            await asyncio.wait(tasks)
            self.assertTrue(all(t.cancelled() for t in tasks))

        asyncio.run(run())

    def test_cancel_error_upstreams_and_gather_error(self) -> None:
        #  A1      B1
        #   |      |
        #  A2      B2 (raises)
        #    \    /
        #      C1

        async def run() -> None:
            a1 = _node("A1", [])
            a2 = _node("A2", [a1])
            b1 = _node("B1", [])
            b2 = _node("B2", [b1], exc=DummyException("fail B2"))
            c1 = _node("C1", [a2, b2])
            tasks = _start_tasks(c1)
            await asyncio.sleep(0)

            # Let B2 fail
            await asyncio.wait([b2.task])

            _cancel_orphaned(c1)
            await asyncio.wait(tasks)

            # Only B1 should be cancelled, since B2 errored
            self.assertFalse(a1.task.cancelled())
            self.assertFalse(a2.task.cancelled())
            self.assertTrue(b1.task.cancelled())
            self.assertFalse(b2.task.cancelled())
            self.assertFalse(c1.task.cancelled())

            errs = _gather_error(c1)
            self.assertEqual(len(errs), 1)
            name, err = errs[0]
            self.assertEqual(name, "0:0:B2")
            self.assertIsInstance(err, DummyException)
            self.assertEqual(err.args[0], "fail B2")

        asyncio.run(run())

    def test_cancel_error_upstreams_and_gather_error_multiple(self) -> None:
        #          A1      B1
        #           |      |
        # (raises) A2      B2 (raises)
        #            \    /
        #              C1
        async def run() -> None:
            a1 = _node("A1", [])
            a2 = _node("A2", [a1], exc=DummyException("fail A2"))
            b1 = _node("B1", [])
            b2 = _node("B2", [b1], exc=DummyException("fail B2"))
            c1 = _node("C1", [a2, b2])

            tasks = _start_tasks(c1)
            await asyncio.sleep(0)

            # Let A2, B2 fail
            await asyncio.wait([a2.task, b2.task])

            _cancel_orphaned(c1)
            await asyncio.wait(tasks)

            self.assertTrue(a1.task.cancelled())
            self.assertFalse(a2.task.cancelled())
            self.assertTrue(b1.task.cancelled())
            self.assertFalse(b2.task.cancelled())
            self.assertFalse(c1.task.cancelled())

            errs = _gather_error(c1)
            self.assertEqual(len(errs), 2)
            name, err = errs[0]
            self.assertEqual(name, "0:0:A2")
            self.assertIsInstance(err, DummyException)
            self.assertEqual(err.args[0], "fail A2")
            name, err = errs[1]
            self.assertEqual(name, "0:0:B2")
            self.assertIsInstance(err, DummyException)
            self.assertEqual(err.args[0], "fail B2")

        asyncio.run(run())

    def test_cancel_error_upstreams_and_gather_error_complex(self) -> None:
        #           B1      C1
        #            |      |
        #  (raises) B2      C2 (raises)
        #             \    /
        #          A1   D1  E1
        #           |   |   |
        # (raises) A2   D2  E2
        #            \  |  /
        #               F1

        async def run() -> None:
            a1 = _node("A1", [])
            a2 = _node("A2", [a1], exc=DummyException("fail A2"))
            b1 = _node("B1", [])
            b2 = _node("B2", [b1], exc=DummyException("fail B2"))
            c1 = _node("C1", [])
            c2 = _node("C2", [c1], exc=DummyException("fail C2"))
            d1 = _node("D1", [b2, c2])
            d2 = _node("D2", [d1])
            e1 = _node("E1", [])
            e2 = _node("E1", [e1])
            f1 = _node("F1", [a2, d2, e2])

            tasks = _start_tasks(f1)
            await asyncio.sleep(0)

            # Let A2, B2, C2 fail
            await asyncio.wait([a2.task, b2.task, c2.task])

            _cancel_orphaned(f1)

            # Let the cancellations propagate
            await asyncio.sleep(0)

            self.assertTrue(a1.task.cancelled())
            self.assertFalse(a2.task.cancelled())
            self.assertTrue(b1.task.cancelled())
            self.assertFalse(b2.task.cancelled())
            self.assertTrue(c1.task.cancelled())
            self.assertFalse(c2.task.cancelled())
            self.assertFalse(d1.task.cancelled())
            self.assertFalse(d2.task.cancelled())
            self.assertFalse(e1.task.cancelled())
            self.assertFalse(e2.task.cancelled())
            self.assertFalse(f1.task.cancelled())

            await asyncio.wait(tasks)

            errs = _gather_error(f1)
            self.assertEqual(len(errs), 3)
            name, err = errs[0]
            self.assertEqual(name, "0:0:A2")
            self.assertIsInstance(err, DummyException)
            self.assertEqual(err.args[0], "fail A2")
            name, err = errs[1]
            self.assertEqual(name, "0:0:B2")
            self.assertIsInstance(err, DummyException)
            self.assertEqual(err.args[0], "fail B2")
            name, err = errs[2]
            self.assertEqual(name, "0:0:C2")
            self.assertIsInstance(err, DummyException)
            self.assertEqual(err.args[0], "fail C2")

        asyncio.run(run())

    def test_gather_error_with_cancelled(self) -> None:
        #  A1      B1
        #   |      |
        #  A2      B2
        #    \    /
        #      C1 (raises)

        async def run() -> None:
            a1 = _node("A1", [])
            a2 = _node("A2", [a1])
            b1 = _node("B1", [])
            b2 = _node("B2", [b1])
            c1 = _node("C1", [a2, b2], exc=DummyException("fail C"))

            tasks = _start_tasks(c1)
            await asyncio.sleep(0)

            # Let C fail
            await asyncio.wait([c1.task])

            _cancel_orphaned(c1)
            await asyncio.wait(tasks)

            self.assertTrue(a1.task.cancelled())
            self.assertTrue(a2.task.cancelled())
            self.assertTrue(b1.task.cancelled())
            self.assertTrue(b2.task.cancelled())
            self.assertFalse(c1.task.cancelled())

            errs = _gather_error(c1)
            self.assertEqual(len(errs), 1)
            name, err = errs[0]
            self.assertEqual(name, "0:0:C1")
            self.assertIsInstance(err, DummyException)

        asyncio.run(run())
