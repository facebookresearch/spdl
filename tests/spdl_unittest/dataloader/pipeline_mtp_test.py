# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import multiprocessing as mp
import time
from collections.abc import Iterable, Iterator
from queue import Empty

import pytest
from spdl.pipeline._utils import (
    _execute_iterator,
    _MSG_GENERATOR_FAILED,
    _MSG_INITIALIZER_FAILED,
    _MSG_ITERATION_FINISHED,
    _MSG_PARENT_REQUEST_STOP,
)

#######################################################################################
# iterate_in_pipeline
#######################################################################################


class SourceIterable:
    def __init__(self, n: int) -> None:
        self.n = n

    def __iter__(self) -> Iterator[int]:
        yield from range(self.n)


def noop() -> None:
    pass


def test_execute_iterator_initializer_failure():
    msg_queue, data_queue = mp.Queue(), mp.Queue()

    def src_fn() -> Iterable[int]:
        return SourceIterable(10)

    def fail() -> None:
        raise ValueError("Failed!")

    with pytest.raises(ValueError):
        _execute_iterator(msg_queue, data_queue, src_fn, fail)

    assert msg_queue.get(timeout=1) == _MSG_INITIALIZER_FAILED
    assert msg_queue.empty()
    assert data_queue.empty()


def test_execute_iterator_iterator_initialize_failure():
    msg_queue, data_queue = mp.Queue(), mp.Queue()

    def src_fn() -> Iterator[int]:
        raise ValueError("Failed!")
        return SourceIterable(10)

    with pytest.raises(ValueError):
        _execute_iterator(msg_queue, data_queue, src_fn, noop)

    assert msg_queue.get(timeout=1) == _MSG_GENERATOR_FAILED
    assert msg_queue.empty()
    assert data_queue.empty()


def test_execute_iterator_quite_immediately():
    msg_queue, data_queue = mp.Queue(), mp.Queue()

    msg_queue.put(_MSG_PARENT_REQUEST_STOP)
    time.sleep(1)

    def src_fn() -> Iterable[int]:
        return SourceIterable(10)

    _execute_iterator(msg_queue, data_queue, src_fn, noop)

    assert msg_queue.empty()
    assert data_queue.empty()


def test_execute_iterator_generator_fail():
    msg_queue, data_queue = mp.Queue(), mp.Queue()

    class SourceIterableFails(SourceIterable):
        def __iter__(self) -> Iterator[int]:
            raise ValueError("Failed!")
            yield from range(self.n)

    def src_fn() -> Iterable[int]:
        return SourceIterableFails(10)

    _execute_iterator(msg_queue, data_queue, src_fn, noop)

    assert msg_queue.get(timeout=1) == _MSG_GENERATOR_FAILED
    assert msg_queue.empty()
    assert data_queue.empty()


def test_execute_iterator_generator_fail_after_n():
    msg_queue, data_queue = mp.Queue(), mp.Queue()

    class SourceIterableFails(SourceIterable):
        def __iter__(self) -> Iterator[int]:
            for v in range(self.n):
                yield v
                if v == 2:
                    raise ValueError("Failed!")

    def src_fn() -> Iterable[int]:
        return SourceIterableFails(10)

    _execute_iterator(msg_queue, data_queue, src_fn, noop)

    assert data_queue.get(timeout=1) == 0
    assert data_queue.get(timeout=1) == 1
    assert data_queue.get(timeout=1) == 2

    with pytest.raises(Empty):
        data_queue.get(timeout=1)

    assert msg_queue.get(timeout=1) == _MSG_GENERATOR_FAILED
    assert msg_queue.empty()


def test_execute_iterator_generator_success():
    msg_queue, data_queue = mp.Queue(), mp.Queue()

    def src_fn() -> Iterable[int]:
        return SourceIterable(3)

    _execute_iterator(msg_queue, data_queue, src_fn, noop)

    assert data_queue.get(timeout=1) == 0
    assert data_queue.get(timeout=1) == 1
    assert data_queue.get(timeout=1) == 2

    with pytest.raises(Empty):
        data_queue.get(timeout=1)

    assert msg_queue.get(timeout=1) == _MSG_ITERATION_FINISHED
    assert msg_queue.empty()
