# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

from spdl.pipeline._node import (
    _cancel_recursive,
    _cancel_upstreams_of_errors,
    _gather_error,
    _Node,
    _start_tasks,
)
from spdl.pipeline._queue import AsyncQueue

# pyre-strict


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

    return _Node(name, coro(), AsyncQueue(name), deps)


def test_node_chain_start_and_cancel() -> None:
    #  A -> B -> C

    async def run() -> None:
        a = _node("A", [])
        b = _node("B", [a])
        c = _node("C", [b])
        tasks = _start_tasks(c)
        assert all(isinstance(t, asyncio.Task) for t in tasks)
        assert len(tasks) == 3

        canceled = _cancel_recursive(c)
        assert canceled == tasks
        await asyncio.gather(*tasks, return_exceptions=True)
        assert all(t.cancelled() for t in tasks)

    asyncio.run(run())


def test_node_y_shape_upstream() -> None:
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
        assert len(tasks) == 5

        canceled = _cancel_recursive(c1)
        assert canceled == tasks
        await asyncio.gather(*tasks, return_exceptions=True)
        assert all(t.cancelled() for t in tasks)

    asyncio.run(run())


def test_cancel_error_upstreams_and_gather_error() -> None:
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
        await asyncio.sleep(0)  # Let tasks start

        # Let B2 fail
        await asyncio.gather(b2.task, return_exceptions=True)

        canceled = _cancel_upstreams_of_errors(c1)
        # Only B1 should be canceled, since B2 errored
        assert a1.task not in canceled
        assert a2.task not in canceled
        assert b1.task in canceled
        assert b2.task not in canceled
        assert c1.task not in canceled

        errs = _gather_error(c1)
        assert len(errs) == 1
        name, err = errs[0]
        assert name == "B2"
        assert isinstance(err, DummyException) and err.args[0] == "fail B2"
        await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run())


def test_cancel_error_upstreams_and_gather_error_multiple() -> None:
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
        await asyncio.sleep(0)  # Let tasks start

        # Let A2, B2 fail
        await asyncio.gather(a2.task, b2.task, return_exceptions=True)

        canceled = _cancel_upstreams_of_errors(c1)
        assert a1.task in canceled
        assert a2.task not in canceled
        assert b1.task in canceled
        assert b2.task not in canceled
        assert c1.task not in canceled

        errs = _gather_error(c1)
        assert len(errs) == 2
        name, err = errs[0]
        assert name == "A2"
        assert isinstance(err, DummyException) and err.args[0] == "fail A2"
        name, err = errs[1]
        assert name == "B2"
        assert isinstance(err, DummyException) and err.args[0] == "fail B2"

        await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run())


def test_cancel_error_upstreams_and_gather_error_complex() -> None:
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
        await asyncio.sleep(0)  # Let tasks start

        # Let A2, B2, C2 fail
        await asyncio.gather(a2.task, b2.task, c2.task, return_exceptions=True)

        canceled = _cancel_upstreams_of_errors(f1)
        assert a1.task in canceled
        assert a2.task not in canceled
        assert b1.task in canceled
        assert b2.task not in canceled
        assert c1.task in canceled
        assert c2.task not in canceled
        assert d1.task not in canceled
        assert d2.task not in canceled
        assert e1.task not in canceled
        assert e2.task not in canceled
        assert f1.task not in canceled

        errs = _gather_error(f1)
        assert len(errs) == 3
        name, err = errs[0]
        assert name == "A2"
        assert isinstance(err, DummyException) and err.args[0] == "fail A2"
        name, err = errs[1]
        assert name == "B2"
        assert isinstance(err, DummyException) and err.args[0] == "fail B2"
        name, err = errs[2]
        assert name == "C2"
        assert isinstance(err, DummyException) and err.args[0] == "fail C2"

        await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run())


def test_gather_error_with_cancelled() -> None:
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
        await asyncio.sleep(0)  # Let tasks start

        # Let C fail
        await asyncio.gather(c1.task, return_exceptions=True)

        canceled = _cancel_upstreams_of_errors(c1)
        assert a1.task in canceled
        assert a2.task in canceled
        assert b1.task in canceled
        assert b2.task in canceled
        assert c1.task not in canceled

        errs = _gather_error(c1)
        assert len(errs) == 1
        name, err = errs[0]
        assert name == "C1"
        assert isinstance(err, DummyException)
        await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run())
