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
import threading
import time
import unittest
from collections.abc import Iterable, Iterator
from functools import partial
from unittest.mock import patch

from spdl.pipeline import iterate_in_subprocess as _iterate_in_subprocess
from spdl.pipeline._iter_utils._common import _Cmd, _execute_iterable, _Status
from spdl.source.utils import (
    embed_shuffle,
    IterableWithShuffle,
    MergeIterator,
    repeat_source,
)


def iterate_in_subprocess(fn, *, timeout=10, **kwargs):
    return _iterate_in_subprocess(fn, timeout=timeout, **kwargs)


class TestMergeIterator(unittest.TestCase):
    def test_mergeiterator_ordered(self) -> None:
        """MergeIterator iterates multiple iterators"""

        iterables = [
            [0, 1, 2],
            [10, 11, 12],
            [20, 21, 22],
        ]

        result = list(MergeIterator(iterables))
        self.assertEqual(result, [0, 10, 20, 1, 11, 21, 2, 12, 22])

    def test_mergeiterator_ordered_stop_after_first_exhaustion(self) -> None:
        """MergeIterator stops after the first exhaustion"""

        iterables = [
            [0],
            [10, 11, 12],
            [20, 21, 22],
        ]

        result = list(MergeIterator(iterables, stop_after=-1))
        self.assertEqual(result, [0, 10, 20])

        iterables = [
            [0, 1, 2],
            [10],
            [20, 21, 22],
        ]

        result = list(MergeIterator(iterables, stop_after=-1))
        self.assertEqual(result, [0, 10, 20, 1])

        iterables = [
            [0, 1, 2],
            [10, 11],
            [20],
        ]

        result = list(MergeIterator(iterables, stop_after=-1))
        self.assertEqual(result, [0, 10, 20, 1, 11])

    def test_mergeiterator_ordered_stop_after_N(self) -> None:
        """MergeIterator stops after N items are yielded"""

        iterables = [
            [0, 1, 2],
            [10, 11, 12],
            [20, 21, 22],
        ]

        result = list(MergeIterator(iterables, stop_after=1))
        self.assertEqual(result, [0])

        result = list(MergeIterator(iterables, stop_after=5))
        self.assertEqual(result, [0, 10, 20, 1, 11])

        result = list(MergeIterator(iterables, stop_after=7))
        self.assertEqual(result, [0, 10, 20, 1, 11, 21, 2])

    def test_mergeiterator_ordered_stop_after_minus1(self) -> None:
        """MergeIterator stops after all the iterables are exhausted"""

        iterables = [
            [0, 1, 2],
            [10, 11, 12],
            [20, 21, 22],
        ]

        result = list(MergeIterator(iterables))
        self.assertEqual(result, [0, 10, 20, 1, 11, 21, 2, 12, 22])

        iterables = [
            [0, 1, 2],
            [10],
            [20, 21, 22],
        ]

        result = list(MergeIterator(iterables))
        self.assertEqual(result, [0, 10, 20, 1, 21, 2, 22])

        iterables = [
            [0, 1, 2],
            [10, 11, 12],
            [20],
        ]

        result = list(MergeIterator(iterables))
        self.assertEqual(result, [0, 10, 20, 1, 11, 2, 12])

    def test_mergeiterator_ordered_n(self) -> None:
        """with stop_after=N, MergeIterator continues iterating after exhaustion."""
        iterables = [
            [0, 1, 2],
            [10],
            [20, 21, 22],
        ]

        result = list(MergeIterator(iterables, stop_after=5))
        self.assertEqual(result, [0, 10, 20, 1, 21])

        result = list(MergeIterator(iterables, stop_after=7))
        self.assertEqual(result, [0, 10, 20, 1, 21, 2, 22])

        result = list(MergeIterator(iterables, stop_after=8))
        self.assertEqual(result, [0, 10, 20, 1, 21, 2, 22])

    def test_mergeiterator_stochastic_smoke_test(self) -> None:
        """MergeIterator with probabilitiies do not get stuck."""

        iterables = [
            [0, 1, 2],
            [10, 11, 12],
            [20, 21, 22],
        ]

        weights = [1, 1, 1]

        result = list(MergeIterator(iterables, weights=weights))
        self.assertEqual(set(result), {0, 1, 2, 10, 11, 12, 20, 21, 22})

    def test_mergeiterator_stochastic_rejects_zero(self) -> None:
        """weight=0 is rejected."""
        weights = [1, 0]

        with self.assertRaises(ValueError):
            MergeIterator([[1]], weights=weights)

        weights = [1, 0.0]

        with self.assertRaises(ValueError):
            MergeIterator([[1]], weights=weights)

    def test_mergeiterator_skip_zero_weight(self) -> None:
        """Iterables with zero weight are skipped."""
        iterables = [
            [0, 1, 2],
            [10, 11, 12],
            [20, 21, 22],
            [30, 31, 32],
        ]

        weights = [1, 0, 2, 0]

        merge_iter = MergeIterator(iterables, weights=weights)

        self.assertEqual(len(merge_iter.iterables), 2)
        self.assertEqual(merge_iter.iterables[0], [0, 1, 2])
        self.assertEqual(merge_iter.iterables[1], [20, 21, 22])

        self.assertIsNotNone(merge_iter.weights)
        # pyre-ignore[16]: weights is not None after assertion
        self.assertEqual(len(merge_iter.weights), 2)
        # pyre-ignore[16]: weights is not None after assertion
        self.assertEqual(merge_iter.weights[0], 1)
        # pyre-ignore[16]: weights is not None after assertion
        self.assertEqual(merge_iter.weights[1], 2)

        result = list(merge_iter)
        self.assertEqual(set(result), {0, 1, 2, 20, 21, 22})

    def test_mergeiterator_stochastic_stop_after_N(self) -> None:
        """Values are taken from iterables with higher weights"""
        weights = [1000000, 1]

        iterables = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        ]

        result = list(MergeIterator(iterables, weights=weights, stop_after=3))
        self.assertEqual(result, [0, 1, 2])

    def test_mergeiterator_stochastic_stop_after_first_exhaustion(self) -> None:
        """Values are taken from iterables with higher weights"""
        weights = [1000000, 1]

        iterables = [
            [0, 1, 2, 3],
            [10, 11, 12, 13],
        ]

        result = list(MergeIterator(iterables, weights=weights, stop_after=-1))
        self.assertEqual(result, [0, 1, 2, 3])


class TestRepeatSource(unittest.TestCase):
    def test_repeat_source_iterable_with_shuffle(self) -> None:
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
            self.assertEqual(next(gen), 1)
            mock_method.assert_called_with(seed=0)
            self.assertEqual(next(gen), 2)
            self.assertEqual(next(gen), 0)

            self.assertEqual(next(gen), 2)
            mock_method.assert_called_with(seed=1)
            self.assertEqual(next(gen), 0)
            self.assertEqual(next(gen), 1)

            self.assertEqual(next(gen), 0)
            mock_method.assert_called_with(seed=2)
            self.assertEqual(next(gen), 1)
            self.assertEqual(next(gen), 2)

            self.assertEqual(next(gen), 1)
            mock_method.assert_called_with(seed=3)
            self.assertEqual(next(gen), 2)
            self.assertEqual(next(gen), 0)

            self.assertEqual(next(gen), 2)
            mock_method.assert_called_with(seed=4)
            self.assertEqual(next(gen), 0)
            self.assertEqual(next(gen), 1)

    def test_repeat_source_iterable(self) -> None:
        """repeat_source works Iterable without shuffle method"""

        class _IteWithoutShuffle:
            def __init__(self) -> None:
                self.vals = list(range(3))

            def __iter__(self) -> Iterator[int]:
                yield from self.vals

        src = _IteWithoutShuffle()
        gen = iter(repeat_source(src, epoch=2))

        for _ in range(100):
            self.assertEqual(next(gen), 0)
            self.assertEqual(next(gen), 1)
            self.assertEqual(next(gen), 2)

    def test_repeat_source_picklable(self) -> None:
        """repeat_source is picklable."""

        src = list(range(10))
        src = repeat_source(src)

        serialized = pickle.dumps(src)
        src2 = pickle.loads(serialized)

        for _ in range(3):
            for i in range(10):
                self.assertEqual(next(src), i)
                self.assertEqual(next(src2), i)


#######################################################################################
# iterate_in_pipeline
#######################################################################################


def iter_range(n: int) -> Iterable[int]:
    yield from range(n)


def initializer(path: str, val: str) -> None:
    with open(path, "w") as f:
        f.write(val)


class TestIterateInSubprocess(unittest.TestCase):
    def test_iterate_in_subprocess(self) -> None:
        """iterate_in_subprocess iterates"""
        N = 10

        src = iterate_in_subprocess(fn=partial(iter_range, n=N))
        self.assertEqual(list(src), list(range(N)))

    def test_iterate_in_subprocess_initializer(self) -> None:
        """iterate_in_subprocess initializer is called before iteration starts"""

        N = 10
        val = str(random.random())
        with tempfile.TemporaryDirectory() as dir:
            path = os.path.join(dir, "foo.txt")

            self.assertFalse(os.path.exists(path))
            src = iterate_in_subprocess(
                fn=partial(iter_range, n=N),
                initializer=partial(initializer, path=path, val=val),
                buffer_size=1,
            )
            self.assertTrue(os.path.exists(path))

            ite = iter(src)
            self.assertEqual(next(ite), 0)

            with open(path, "r") as f:
                self.assertEqual(f.read(), val)

        for i in range(1, N):
            self.assertEqual(next(ite), i)

        with self.assertRaises(StopIteration):
            next(ite)

    def test_iterate_in_subprocess_multiple_initializer(self) -> None:
        """iterate_in_subprocess accepts multiple initializers"""
        N = 10
        val1 = str(random.random())
        val2 = str(random.random())
        with tempfile.TemporaryDirectory() as dir:
            path1 = os.path.join(dir, "foo.txt")
            path2 = os.path.join(dir, "bar.txt")

            self.assertFalse(os.path.exists(path1))
            self.assertFalse(os.path.exists(path2))
            src = iterate_in_subprocess(
                fn=partial(iter_range, n=N),
                initializer=[
                    partial(initializer, path=path1, val=val1),
                    partial(initializer, path=path2, val=val2),
                ],
                buffer_size=1,
            )
            self.assertTrue(os.path.exists(path1))
            self.assertTrue(os.path.exists(path2))

            ite = iter(src)
            self.assertEqual(next(ite), 0)

            with open(path1, "r") as f:
                self.assertEqual(f.read(), val1)

            with open(path2, "r") as f:
                self.assertEqual(f.read(), val2)

        for i in range(1, N):
            self.assertEqual(next(ite), i)

        with self.assertRaises(StopIteration):
            next(ite)


def iter_range_and_store_with_sync(n: int, sync_queue: mp.Queue) -> Iterable[int]:
    """Generator that synchronizes with main process via queue."""
    yield 0
    for i in range(n):
        yield i
        # Signal main process that we've yielded this value
        sync_queue.put(i)


class TestIterateInSubprocessBufferSize(unittest.TestCase):
    def test_iterate_in_subprocess_buffer_size_1(self) -> None:
        """buffer_size=1 makes iterate_in_subprocess works sort-of interactively"""

        N = 10

        # Use queue for synchronization between processes
        sync_queue = mp.Queue()

        src = iterate_in_subprocess(
            fn=partial(iter_range_and_store_with_sync, n=N, sync_queue=sync_queue),
            daemon=True,
            buffer_size=1,
        )
        ite = iter(src)
        self.assertEqual(next(ite), 0)

        for i in range(N):
            # Wait for subprocess to signal it has yielded value i
            # Use timeout to avoid hanging if subprocess fails
            subprocess_value = sync_queue.get(timeout=5)
            self.assertEqual(
                subprocess_value, i, f"Expected {i}, got {subprocess_value}"
            )

            # With buffer_size=1, the queue should be empty after fetching
            self.assertTrue(
                sync_queue.empty(), f"Queue should be empty after fetching item {i}"
            )

            # Now fetch the value from the iterator
            self.assertEqual(next(ite), i)

        with self.assertRaises(StopIteration):
            next(ite)

    def test_iterate_in_subprocess_buffer_size_64(self) -> None:
        """big buffer_size makes iterate_in_subprocess processes data in one go"""

        N = 10

        # Use queue for synchronization between processes
        sync_queue = mp.Queue()

        src = iterate_in_subprocess(
            fn=partial(iter_range_and_store_with_sync, n=N, sync_queue=sync_queue),
            daemon=True,
            buffer_size=64,
        )
        ite = iter(src)
        self.assertEqual(next(ite), 0)

        # With buffer_size=64, subprocess should process all data without waiting
        # Wait for subprocess to signal all values have been processed
        for expected_i in range(N):
            # Use timeout to avoid hanging
            subprocess_value = sync_queue.get(timeout=5)
            self.assertEqual(
                subprocess_value,
                expected_i,
                f"Expected {expected_i}, got {subprocess_value}",
            )

        # Now all data should be available in the buffer, fetch them
        for i in range(N):
            self.assertEqual(next(ite), i)

        with self.assertRaises(StopIteration):
            next(ite)


class SourceIterable:
    def __init__(self, n: int) -> None:
        self.n = n

    def __iter__(self) -> Iterator[int]:
        yield from range(self.n)


def noop() -> None:
    pass


class TestExecuteIterable(unittest.TestCase):
    def test_execute_iterable_initializer_failure(self) -> None:
        msg_queue, data_queue = mp.Queue(), mp.Queue()

        def src_fn() -> Iterable[int]:
            return SourceIterable(10)

        def fail() -> None:
            raise ValueError("Failed!")

        _execute_iterable(msg_queue, data_queue, src_fn, [fail])

        self.assertTrue(msg_queue.empty())

        result = data_queue.get(timeout=1)
        self.assertEqual(result.status, _Status.INITIALIZATION_FAILED)
        self.assertIn("Failed!", result.message)
        self.assertTrue(data_queue.empty())

    def test_execute_iterable_iterator_initialize_failure(self) -> None:
        msg_queue, data_queue = mp.Queue(), mp.Queue()

        def src_fn() -> Iterator[int]:
            raise ValueError("Failed!")
            return SourceIterable(10)

        _execute_iterable(msg_queue, data_queue, src_fn, [noop])

        self.assertTrue(msg_queue.empty())
        result = data_queue.get(timeout=1)
        self.assertEqual(result.status, _Status.INITIALIZATION_FAILED)
        self.assertIn("Failed!", result.message)
        self.assertTrue(data_queue.empty())

    def test_execute_iterable_quite_immediately(self) -> None:
        msg_queue, data_queue = mp.Queue(), mp.Queue()

        msg_queue.put(_Cmd.ABORT)
        time.sleep(1)

        def src_fn() -> Iterable[int]:
            return SourceIterable(10)

        _execute_iterable(msg_queue, data_queue, src_fn, [noop])
        time.sleep(1)

        self.assertTrue(msg_queue.empty())
        ack = data_queue.get(timeout=1)
        self.assertEqual(ack.status, _Status.INITIALIZATION_SUCCEEDED)
        self.assertTrue(data_queue.empty())

    def test_execute_iterable_generator_fail(self) -> None:
        msg_queue, data_queue = mp.Queue(), mp.Queue()

        class SourceIterableFails(SourceIterable):
            def __iter__(self) -> Iterator[int]:
                raise ValueError("Failed!")
                yield from range(self.n)

        def src_fn() -> Iterable[int]:
            return SourceIterableFails(10)

        msg_queue.put(_Cmd.START_ITERATION)
        _execute_iterable(msg_queue, data_queue, src_fn, [noop])

        self.assertTrue(msg_queue.empty())

        ack = data_queue.get(timeout=1)
        self.assertEqual(ack.status, _Status.INITIALIZATION_SUCCEEDED)
        ack = data_queue.get(timeout=1)
        self.assertEqual(ack.status, _Status.ITERATION_STARTED)

        result = data_queue.get(timeout=1)
        self.assertEqual(result.status, _Status.ITERATOR_FAILED)
        self.assertIn("Failed!", result.message)
        self.assertTrue(data_queue.empty())

    def test_execute_iterable_generator_fail_after_n(self) -> None:
        msg_queue, data_queue = mp.Queue(), mp.Queue()

        class SourceIterableFails(SourceIterable):
            def __iter__(self) -> Iterator[int]:
                for v in range(self.n):
                    yield v
                    if v == 2:
                        raise ValueError("Failed!")

        def src_fn() -> Iterable[int]:
            return SourceIterableFails(10)

        msg_queue.put(_Cmd.START_ITERATION)
        _execute_iterable(msg_queue, data_queue, src_fn, [noop])

        self.assertTrue(msg_queue.empty())

        ack = data_queue.get(timeout=1)
        self.assertEqual(ack.status, _Status.INITIALIZATION_SUCCEEDED)
        ack = data_queue.get(timeout=1)
        self.assertEqual(ack.status, _Status.ITERATION_STARTED)
        for i in range(3):
            result = data_queue.get(timeout=1)
            self.assertEqual(result.status, _Status.ITERATOR_SUCCESS)
            self.assertEqual(result.message, i)

        result = data_queue.get(timeout=1)
        self.assertEqual(result.status, _Status.ITERATOR_FAILED)
        self.assertIn("Failed!", result.message)
        self.assertTrue(data_queue.empty())

    def test_execute_iterator_generator_success(self) -> None:
        msg_queue, data_queue = mp.Queue(), mp.Queue()

        def src_fn() -> Iterable[int]:
            return SourceIterable(3)

        msg_queue.put(_Cmd.START_ITERATION)

        # Add abort with delay, so that _execute_iterable can exit after
        # the iteration
        def done():
            time.sleep(3)
            msg_queue.put(_Cmd.ABORT)

        t = threading.Thread(target=done)
        t.start()
        _execute_iterable(msg_queue, data_queue, src_fn, [noop])
        t.join()

        self.assertTrue(msg_queue.empty())

        ack = data_queue.get(timeout=1)
        self.assertEqual(ack.status, _Status.INITIALIZATION_SUCCEEDED)
        ack = data_queue.get(timeout=1)
        self.assertEqual(ack.status, _Status.ITERATION_STARTED)
        for i in range(3):
            result = data_queue.get(timeout=1)
            self.assertEqual(result.status, _Status.ITERATOR_SUCCESS)
            self.assertEqual(result.message, i)

        result = data_queue.get(timeout=1)
        self.assertEqual(result.status, _Status.ITERATION_FINISHED)


def _src1() -> Iterable[int]:
    return SourceIterable(10)


def _init1() -> None:
    raise ValueError("Failed!")


def _src2() -> Iterator[int]:
    if True:
        raise ValueError("Failed!")
    return SourceIterable(10)


def _src3() -> Iterable[int]:
    class SourceIterableFails(SourceIterable):
        def __iter__(self) -> Iterator[int]:
            raise ValueError("Failed!")
            yield from range(self.n)

    return SourceIterableFails(10)


def _src4() -> Iterable[int]:
    class SourceIterableFails(SourceIterable):
        def __iter__(self) -> Iterator[int]:
            for v in range(self.n):
                yield v
                if v == 2:
                    raise ValueError("Failed!")

    return SourceIterableFails(10)


def _src5(N) -> Iterable[int]:
    return SourceIterable(N)


class SleepSourceIterable(SourceIterable):
    def __iter__(self):
        time.sleep(10)
        yield 0


def _src6() -> Iterable[int]:
    return SleepSourceIterable(3)


def _fail_initializer():
    raise RuntimeError("Failed!")


class IterableWithShuffleSource:
    def __init__(self, n: int) -> None:
        self.vals = list(range(n))

    def __iter__(self) -> Iterator[int]:
        yield from self.vals

    def shuffle(self, seed: int) -> None:
        random.seed(seed)
        random.shuffle(self.vals)


class SourceIterableWithShuffle(IterableWithShuffle[int]):
    def __init__(self, n: int) -> None:
        self.i = 0
        self.vals = list(range(n))

    def shuffle(self, seed: int) -> None:
        assert isinstance(seed, int)
        self.vals = self.vals[1:] + self.vals[:1]

    def __iter__(self) -> Iterator[int]:
        yield from self.vals


_VERY_BAD_REFERENCE = None


class TestIterateInSubprocessFailures(unittest.TestCase):
    def test_iterate_in_subprocess_initializer_failure(self) -> None:
        with self.assertRaisesRegex(RuntimeError, r"Initializer failed"):
            iterate_in_subprocess(_src1, buffer_size=1, timeout=3, initializer=_init1)

    def test_iterate_in_subprocess_iterator_initialize_failure(self) -> None:
        with self.assertRaisesRegex(RuntimeError, r"Failed to create the iterable"):
            iterate_in_subprocess(_src2, buffer_size=1, timeout=3)

    def test_iterate_in_subprocess_generator_fail(self) -> None:
        ite = iter(iterate_in_subprocess(_src3, buffer_size=1, timeout=3))

        with self.assertRaisesRegex(RuntimeError, r"Failed to fetch the next item"):
            next(ite)

    def test_iterate_in_subprocess_fail_after_n(self) -> None:
        ite = iter(iterate_in_subprocess(_src4, buffer_size=1, timeout=3))
        self.assertEqual(next(ite), 0)
        self.assertEqual(next(ite), 1)
        self.assertEqual(next(ite), 2)

        with self.assertRaisesRegex(RuntimeError, r"Failed to fetch the next item"):
            next(ite)

    def test_iterate_in_subprocess_success(self) -> None:
        N = 3

        hyp = list(iterate_in_subprocess(partial(_src5, N), buffer_size=-1, timeout=3))
        self.assertEqual(hyp, list(range(N)))

    def test_iterate_in_subprocess_timeout(self) -> None:
        iterable = iterate_in_subprocess(_src6, buffer_size=-1, timeout=3)
        iterator = iter(iterable)
        with self.assertRaisesRegex(
            RuntimeError, r"The worker subprocess did not produce any data for"
        ):
            next(iterator)

    def test_iterate_in_subprocess_initializer_fail(self) -> None:
        """The initialization failure is propagated to the main process"""

        with self.assertRaisesRegex(RuntimeError, r"Initializer failed"):
            iterate_in_subprocess(SourceIterable, initializer=_fail_initializer)

    def test_iterate_in_subprocess_iterable_creation_fail(self) -> None:
        """The initialization failure is propagated to the main process"""

        with self.assertRaisesRegex(RuntimeError, r"Failed to create the iterable"):
            iterate_in_subprocess(SourceIterable)

    def test_iterate_in_subprocess_success_simple_iterable(self) -> None:
        iterator = iterate_in_subprocess(partial(SourceIterable, 3))

        self.assertEqual(list(iterator), [0, 1, 2])
        self.assertEqual(list(iterator), [0, 1, 2])
        self.assertEqual(list(iterator), [0, 1, 2])

    def test_iterate_in_subprocess_fail_not_stuck(self) -> None:
        """An exception does not make Python stack.

        If a (non-daemon) subprocess is launched without a context manager
        that ensures its clean exit, raising an exception while the reference
        to the process object is held causes the Python interpreter to get
        stuck at the exit.

        To avoid this, we register atexit function, which push the ABORT
        command to the command queue, which will be received by the subprocess
        if the subprocess is not shut down. This test ensures that behavior.
        """

        global _VERY_BAD_REFERENCE
        _VERY_BAD_REFERENCE = iterate_in_subprocess(partial(SourceIterable, 3))
        # Now the Python won't exit unless there is a mechanism to
        # terminate the process.
        # See the use of `atexit` of `iterate_in_subprocess` func.


class TestShuffleAndIterate(unittest.TestCase):
    def test_shuffle_and_iterate_picklable(self) -> None:
        """The result of embed_shuffle must be pickable (for multiprocessing)"""

        src = embed_shuffle(IterableWithShuffleSource(10))
        state = pickle.dumps(src)
        src2 = pickle.loads(state)

        # pyre-ignore[16]: embed_shuffle returns an object with src attribute
        self.assertEqual(src.src.vals, src2.src.vals)

    def test_shuffle_and_iterate(self) -> None:
        N = 10

        src = embed_shuffle(IterableWithShuffleSource(N))

        ref = list(range(N))
        for i in range(3):
            random.seed(i)
            random.shuffle(ref)

            hyp = list(src)
            self.assertEqual(hyp, ref)

    def test_move_iterable_to_subprocess_success_iterable_with_shuffle(self) -> None:
        """IterableWithShuffle can be executed in the subprocess."""
        iterator = iterate_in_subprocess(
            partial(embed_shuffle, SourceIterableWithShuffle(3))
        )

        self.assertEqual(list(iterator), [1, 2, 0])
        self.assertEqual(list(iterator), [2, 0, 1])
        self.assertEqual(list(iterator), [0, 1, 2])
