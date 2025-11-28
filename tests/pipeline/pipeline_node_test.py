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
    _cancel_recursive,
    _cancel_upstreams_of_errors,
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
            await asyncio.sleep(0)

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

            canceled = _cancel_recursive(c)
            self.assertEqual(canceled, tasks)
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

            canceled = _cancel_recursive(c1)
            self.assertEqual(canceled, tasks)
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

            canceled = _cancel_upstreams_of_errors(c1)
            self.assertNotIn(a1.task, canceled)
            self.assertNotIn(a2.task, canceled)
            self.assertIn(b1.task, canceled)
            self.assertNotIn(b2.task, canceled)
            self.assertNotIn(c1.task, canceled)

            await asyncio.wait(tasks)

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

            canceled = _cancel_upstreams_of_errors(c1)
            self.assertIn(a1.task, canceled)
            self.assertNotIn(a2.task, canceled)
            self.assertIn(b1.task, canceled)
            self.assertNotIn(b2.task, canceled)
            self.assertNotIn(c1.task, canceled)

            await asyncio.wait(tasks)

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

            canceled = _cancel_upstreams_of_errors(f1)
            self.assertIn(a1.task, canceled)
            self.assertNotIn(a2.task, canceled)
            self.assertIn(b1.task, canceled)
            self.assertNotIn(b2.task, canceled)
            self.assertIn(c1.task, canceled)
            self.assertNotIn(c2.task, canceled)
            self.assertNotIn(d1.task, canceled)
            self.assertNotIn(d2.task, canceled)
            self.assertNotIn(e1.task, canceled)
            self.assertNotIn(e2.task, canceled)
            self.assertNotIn(f1.task, canceled)

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

            canceled = _cancel_upstreams_of_errors(c1)
            self.assertIn(a1.task, canceled)
            self.assertIn(a2.task, canceled)
            self.assertIn(b1.task, canceled)
            self.assertIn(b2.task, canceled)
            self.assertNotIn(c1.task, canceled)

            await asyncio.wait(tasks)

            errs = _gather_error(c1)
            self.assertEqual(len(errs), 1)
            name, err = errs[0]
            self.assertEqual(name, "C1")
            self.assertIsInstance(err, DummyException)

        asyncio.run(run())
