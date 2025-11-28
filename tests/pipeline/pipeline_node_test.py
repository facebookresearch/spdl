# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import unittest

from spdl.pipeline._components import AsyncQueue
from spdl.pipeline._components._node import (
    _cancel_orphaned,
    _cancel_recursive,
    _gather_error,
    _Node,
    _start_tasks,
)
from spdl.pipeline.defs import _ConfigBase


class DummyException(Exception):
    pass


def _node(
    name: str, deps: list[_Node[None]], exc: Exception | None = None
) -> _Node[None]:
    async def coro() -> None:
        if exc:
            raise exc
        else:
            await asyncio.sleep(10)

    n = _Node(name, _ConfigBase(), deps, AsyncQueue(name))
    n._coro = coro()
    return n


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
            self.assertEqual(name, "B2")
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
            self.assertEqual(name, "A2")
            self.assertIsInstance(err, DummyException)
            self.assertEqual(err.args[0], "fail A2")
            name, err = errs[1]
            self.assertEqual(name, "B2")
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
            self.assertEqual(name, "A2")
            self.assertIsInstance(err, DummyException)
            self.assertEqual(err.args[0], "fail A2")
            name, err = errs[1]
            self.assertEqual(name, "B2")
            self.assertIsInstance(err, DummyException)
            self.assertEqual(err.args[0], "fail B2")
            name, err = errs[2]
            self.assertEqual(name, "C2")
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
            self.assertEqual(name, "C1")
            self.assertIsInstance(err, DummyException)

        asyncio.run(run())
