# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import multiprocessing as mp
import os.path
import pickle
import random
import tempfile
import time
from collections.abc import Iterable, Iterator
from functools import partial
from unittest.mock import patch

import pytest
from spdl.pipeline import iterate_in_subprocess
from spdl.pipeline._utils import _Cmd, _execute_iterator, _Status
from spdl.source.utils import MergeIterator, repeat_source


def test_mergeiterator_ordered():
    """MergeIterator iterates multiple iterators"""

    iterables = [
        [0, 1, 2],
        [10, 11, 12],
        [20, 21, 22],
    ]

    result = list(MergeIterator(iterables))
    assert result == [0, 10, 20, 1, 11, 21, 2, 12, 22]


def test_mergeiterator_ordered_stop_after_first_exhaustion():
    """MergeIterator stops after the first exhaustion"""

    iterables = [
        [0],
        [10, 11, 12],
        [20, 21, 22],
    ]

    result = list(MergeIterator(iterables, stop_after=-1))
    assert result == [0, 10, 20]

    iterables = [
        [0, 1, 2],
        [10],
        [20, 21, 22],
    ]

    result = list(MergeIterator(iterables, stop_after=-1))
    assert result == [0, 10, 20, 1]

    iterables = [
        [0, 1, 2],
        [10, 11],
        [20],
    ]

    result = list(MergeIterator(iterables, stop_after=-1))
    assert result == [0, 10, 20, 1, 11]


def test_mergeiterator_ordered_stop_after_N():
    """MergeIterator stops after N items are yielded"""

    iterables = [
        [0, 1, 2],
        [10, 11, 12],
        [20, 21, 22],
    ]

    result = list(MergeIterator(iterables, stop_after=1))
    assert result == [0]

    result = list(MergeIterator(iterables, stop_after=5))
    assert result == [0, 10, 20, 1, 11]

    result = list(MergeIterator(iterables, stop_after=7))
    assert result == [0, 10, 20, 1, 11, 21, 2]


def test_mergeiterator_ordered_stop_after_minus1():
    """MergeIterator stops after all the iterables are exhausted"""

    iterables = [
        [0, 1, 2],
        [10, 11, 12],
        [20, 21, 22],
    ]

    result = list(MergeIterator(iterables))
    assert result == [0, 10, 20, 1, 11, 21, 2, 12, 22]

    iterables = [
        [0, 1, 2],
        [10],
        [20, 21, 22],
    ]

    result = list(MergeIterator(iterables))
    assert result == [0, 10, 20, 1, 21, 2, 22]

    iterables = [
        [0, 1, 2],
        [10, 11, 12],
        [20],
    ]

    result = list(MergeIterator(iterables))
    assert result == [0, 10, 20, 1, 11, 2, 12]


def test_mergeiterator_ordered_n():
    """with stop_after=N, MergeIterator continues iterating after encountering an exhaustion."""
    iterables = [
        [0, 1, 2],
        [10],
        [20, 21, 22],
    ]

    result = list(MergeIterator(iterables, stop_after=5))
    assert result == [0, 10, 20, 1, 21]

    result = list(MergeIterator(iterables, stop_after=7))
    assert result == [0, 10, 20, 1, 21, 2, 22]

    result = list(MergeIterator(iterables, stop_after=8))
    assert result == [0, 10, 20, 1, 21, 2, 22]


def test_mergeiterator_stochastic_smoke_test():
    """MergeIterator with probabilitiies do not get stuck."""

    iterables = [
        [0, 1, 2],
        [10, 11, 12],
        [20, 21, 22],
    ]

    weights = [1, 1, 1]

    result = list(MergeIterator(iterables, weights=weights))
    assert set(result) == {0, 1, 2, 10, 11, 12, 20, 21, 22}


def test_mergeiterator_stochastic_rejects_zero():
    """weight=0 is rejected."""
    weights = [1, 0]

    with pytest.raises(ValueError):
        MergeIterator([[1]], weights=weights)

    weights = [1, 0.0]

    with pytest.raises(ValueError):
        MergeIterator([[1]], weights=weights)


def test_mergeiterator_stochastic_stop_after_N():
    """Values are taken from iterables with higher weights"""
    weights = [1000000, 1]

    iterables = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    ]

    result = list(MergeIterator(iterables, weights=weights, stop_after=3))
    assert result == [0, 1, 2]


def test_mergeiterator_stochastic_stop_after_first_exhaustion():
    """Values are taken from iterables with higher weights"""
    weights = [1000000, 1]

    iterables = [
        [0, 1, 2, 3],
        [10, 11, 12, 13],
    ]

    result = list(MergeIterator(iterables, weights=weights, stop_after=-1))
    assert result == [0, 1, 2, 3]


def test_repeat_source_iterable_with_shuffle():
    """repeat_source repeats source while calling shuffle"""

    class _IteWithShuffle:
        def __init__(self) -> None:
            self.vals = list(range(3))

        def shuffle(self, seed: int) -> None:
            assert isinstance(seed, int)
            self.vals = self.vals[1:] + self.vals[:1]

        def __iter__(self) -> Iterator[int]:
            yield from self.vals

    src = _IteWithShuffle()
    gen = iter(repeat_source(src, epoch=2))

    with patch.object(src, "shuffle", side_effect=src.shuffle) as mock_method:
        assert next(gen) == 1
        mock_method.assert_called_with(seed=2)
        assert next(gen) == 2
        assert next(gen) == 0

        assert next(gen) == 2
        mock_method.assert_called_with(seed=3)
        assert next(gen) == 0
        assert next(gen) == 1

        assert next(gen) == 0
        mock_method.assert_called_with(seed=4)
        assert next(gen) == 1
        assert next(gen) == 2

        assert next(gen) == 1
        mock_method.assert_called_with(seed=5)
        assert next(gen) == 2
        assert next(gen) == 0


def test_repeat_source_iterable():
    """repeat_source works Iterable without shuffle method"""

    class _IteWithoutShuffle:
        def __init__(self) -> None:
            self.vals = list(range(3))

        def __iter__(self) -> Iterator[int]:
            yield from self.vals

    src = _IteWithoutShuffle()
    gen = iter(repeat_source(src, epoch=2))

    for _ in range(100):
        assert next(gen) == 0
        assert next(gen) == 1
        assert next(gen) == 2


def test_repeat_source_picklable():
    """repeat_source is picklable."""

    src = list(range(10))
    src = repeat_source(src)

    serialized = pickle.dumps(src)
    src2 = pickle.loads(serialized)

    for _ in range(3):
        for i in range(10):
            assert next(src) == next(src2) == i


#######################################################################################
# iterate_in_pipeline
#######################################################################################


def iter_range(n: int) -> Iterable[int]:
    yield from range(n)


def test_iterate_in_subprocess():
    """iterate_in_subprocess iterates"""
    N = 10

    src = iterate_in_subprocess(fn=partial(iter_range, n=N))
    assert list(src) == list(range(N))


def initializer(path: str, val: str) -> None:
    with open(path, "w") as f:
        f.write(val)


def test_iterate_in_subprocess_initializer():
    """iterate_in_subprocess initializer is called before iteration starts"""

    N = 10
    val = str(random.random())
    with tempfile.TemporaryDirectory() as dir:
        path = os.path.join(dir, "foo.txt")

        assert not os.path.exists(path)
        src = iterate_in_subprocess(
            fn=partial(iter_range, n=N),
            initializer=partial(initializer, path=path, val=val),
            buffer_size=1,
        )
        assert not os.path.exists(path)

        assert next(src) == 0

        assert os.path.exists(path)

        with open(path, "r") as f:
            assert f.read() == val

    for i in range(1, N):
        assert next(src) == i

    with pytest.raises(StopIteration):
        next(src)


def iter_range_and_store(n: int, path: str) -> Iterable[int]:
    yield 0
    for i in range(n):
        yield i
        with open(path, "w") as f:
            f.write(str(i))


def test_iterate_in_subprocess_buffer_size_1():
    """buffer_size=1 makes iterate_in_subprocess works sort-of interactively"""

    N = 10

    with tempfile.TemporaryDirectory() as dir:
        path = os.path.join(dir, "foo.txt")

        src = iterate_in_subprocess(
            fn=partial(iter_range_and_store, n=N, path=path),
            daemon=True,
            buffer_size=1,
        )
        assert src.send(None) == 0

        for i in range(N):
            time.sleep(0.1)

            with open(path, "r") as f:
                assert int(f.read()) == i

            assert next(src) == i

        with pytest.raises(StopIteration):
            next(src)


def test_iterate_in_subprocess_buffer_size_64():
    """big buffer_size makes iterate_in_subprocess processes data in one go"""

    N = 10

    with tempfile.TemporaryDirectory() as dir:
        path = os.path.join(dir, "foo.txt")

        src = iterate_in_subprocess(
            fn=partial(iter_range_and_store, n=N, path=path),
            daemon=True,
            buffer_size=64,
        )
        assert src.send(None) == 0

        time.sleep(0.1)
        for i in range(N):
            with open(path, "r") as f:
                assert int(f.read()) == 9

            assert next(src) == i

        with pytest.raises(StopIteration):
            next(src)


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

    _execute_iterator(msg_queue, data_queue, src_fn, fail)

    assert msg_queue.empty()

    result = data_queue.get(timeout=1)
    assert result.status == _Status.INITIALIZER_FAILED
    assert "Failed!" in result.message
    assert data_queue.empty()


def test_execute_iterator_iterator_initialize_failure():
    msg_queue, data_queue = mp.Queue(), mp.Queue()

    def src_fn() -> Iterator[int]:
        raise ValueError("Failed!")
        return SourceIterable(10)

    _execute_iterator(msg_queue, data_queue, src_fn, noop)

    assert msg_queue.empty()
    result = data_queue.get(timeout=1)
    assert result.status == _Status.GENERATOR_FAILED
    assert "Failed!" in result.message
    assert data_queue.empty()


def test_execute_iterator_quite_immediately():
    msg_queue, data_queue = mp.Queue(), mp.Queue()

    msg_queue.put(_Cmd.INTERRUPT)
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

    assert msg_queue.empty()
    result = data_queue.get(timeout=1)
    assert result.status == _Status.GENERATOR_FAILED
    assert "Failed!" in result.message
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

    assert msg_queue.empty()
    for i in range(3):
        result = data_queue.get(timeout=1)
        assert result.status == _Status.GENERATOR_SUCCESS
        assert result.message == i

    result = data_queue.get(timeout=1)
    assert result.status == _Status.GENERATOR_FAILED
    assert "Failed!" in result.message
    assert data_queue.empty()


def test_execute_iterator_generator_success():
    msg_queue, data_queue = mp.Queue(), mp.Queue()

    def src_fn() -> Iterable[int]:
        return SourceIterable(3)

    _execute_iterator(msg_queue, data_queue, src_fn, noop)

    assert msg_queue.empty()

    for i in range(3):
        result = data_queue.get(timeout=1)
        assert result.status == _Status.GENERATOR_SUCCESS
        assert result.message == i

    result = data_queue.get(timeout=1)
    assert result.status == _Status.ITERATION_FINISHED


def test_terate_in_subprocess_initializer_failure():
    def src_fn() -> Iterable[int]:
        return SourceIterable(10)

    def fail() -> None:
        raise ValueError("Failed!")

    ite = iterate_in_subprocess(src_fn, buffer_size=1, timeout=3, initializer=fail)

    with pytest.raises(RuntimeError, match=r"the initializer failed"):
        next(ite)


def test_iterate_in_subprocess_iterator_initialize_failure():
    def src_fn() -> Iterator[int]:
        raise ValueError("Failed!")
        return SourceIterable(10)

    ite = iterate_in_subprocess(src_fn, buffer_size=1, timeout=3)

    with pytest.raises(RuntimeError, match=r"the generator failed"):
        next(ite)


def test_iterate_in_subprocess_generator_fail():
    class SourceIterableFails(SourceIterable):
        def __iter__(self) -> Iterator[int]:
            raise ValueError("Failed!")
            yield from range(self.n)

    def src_fn() -> Iterable[int]:
        return SourceIterableFails(10)

    ite = iterate_in_subprocess(src_fn, buffer_size=1, timeout=3)

    with pytest.raises(RuntimeError, match=r"the generator failed"):
        next(ite)


def test_iterate_in_subprocess_fail_after_n():
    N = 10

    class SourceIterableFails(SourceIterable):
        def __iter__(self) -> Iterator[int]:
            for v in range(self.n):
                yield v
                if v == 2:
                    raise ValueError("Failed!")

    def src_fn() -> Iterable[int]:
        return SourceIterableFails(N)

    ite = iterate_in_subprocess(src_fn, buffer_size=1, timeout=3)
    assert next(ite) == 0
    assert next(ite) == 1
    assert next(ite) == 2

    with pytest.raises(RuntimeError, match=r"the generator failed"):
        next(ite)


def test_iterate_in_subprocess_success():
    N = 3

    def src_fn() -> Iterable[int]:
        return SourceIterable(N)

    hyp = list(iterate_in_subprocess(src_fn, buffer_size=-1, timeout=3))
    assert hyp == list(range(N))


class SleepSourceIterable(SourceIterable):
    def __iter__(self):
        time.sleep(10)
        yield 0


def test_iterate_in_subprocess_timeout():
    N = 3

    def src_fn() -> Iterable[int]:
        return SleepSourceIterable(N)

    with pytest.raises(
        RuntimeError, match=r"The worker process did not produce any data for"
    ):
        next(iterate_in_subprocess(src_fn, buffer_size=-1, timeout=3))
