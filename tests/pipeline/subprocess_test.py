# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import multiprocessing as mp
import os.path
import random
import tempfile
import threading
import time
import unittest
from collections.abc import Iterable, Iterator
from functools import partial

from spdl.pipeline import iterate_in_subprocess as _iterate_in_subprocess
from spdl.pipeline._iter_utils._common import _Cmd, _execute_iterable, _Status


def iterate_in_subprocess(fn, *, timeout=10, **kwargs):
    return _iterate_in_subprocess(fn, timeout=timeout, **kwargs)


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
        """iterate_in_subprocess accepts multiple iterators"""
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
