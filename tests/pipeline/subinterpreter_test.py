# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import sys
import unittest
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Generic, TypeVar

from parameterized import parameterized
from spdl.pipeline import PipelineBuilder, run_pipeline_in_subinterpreter
from spdl.pipeline._components import _get_global_id, _set_global_id
from spdl.pipeline._iter_utils._subinterpreter import (
    iterate_in_subinterpreter as _iterate_in_subinterpreter,
)

T = TypeVar("T")


def iterate_in_subinterpreter(
    fn: Callable[[], Iterable[T]],
    *,
    buffer_size: int = 3,
    initializer: Callable[[], None] | Sequence[Callable[[], None]] | None = None,
    timeout: float = 5,
) -> Iterable[T]:
    """Set timeout for unittest"""
    return _iterate_in_subinterpreter(
        fn, buffer_size=buffer_size, initializer=initializer, timeout=timeout
    )


class _Wrap(Generic[T]):
    """Helper class to wrap an iterable as a callable.

    This class wraps an iterable object and makes it callable, which is useful
    for testing iterate_in_subinterpreter. It can optionally execute a pre-flight
    function before returning the iterable, allowing for assertions to verify
    that initializers have run correctly in the subinterpreter.

    Args:
        obj: The iterable object to wrap.
        pre: Optional callable to execute before returning the iterable.
            Typically used for assertions in tests.
    """

    def __init__(self, obj: Iterable[T], pre: Callable[[], None] | None = None) -> None:
        self.obj = obj
        self.pre = pre

    def __call__(self) -> Iterable[T]:
        if self.pre is not None:
            self.pre()
        return self.obj


_FLAGS: list[int] = []


def _init_flag0() -> None:
    _FLAGS.append(0)


def _init_flag1() -> None:
    _FLAGS.append(1)


def _check_flag0and1() -> None:
    ref = [0, 1]
    assert _FLAGS == ref, f"{_FLAGS=} != {ref=}"


if sys.version_info >= (3, 14):

    class TestIterateInSubinterpreter(unittest.TestCase):
        """Test cases for iterate_in_subinterpreter function."""

        @parameterized.expand(
            [
                ("basic_iteration", list(range(5))),
                ("string_iteration", ["hello", "world", "test"]),
                ("empty_iterator", []),
            ],
        )
        def test_iteration(self, name: str, ref: list[object]) -> None:  # noqa: ARG002
            """Test iteration with various input types."""
            iterable = iterate_in_subinterpreter(_Wrap(ref))
            result = list(iterable)
            self.assertEqual(result, ref)
            result2 = list(iterable)
            self.assertEqual(result2, ref)

        def test_buffer_size(self) -> None:
            """Test with custom buffer size."""
            ref = list(range(10))
            result = list(iterate_in_subinterpreter(_Wrap(ref), buffer_size=5))
            self.assertEqual(result, ref)

        def test_with_initializers(self) -> None:
            """Test with multiple initializer functions."""
            ref = list(range(10))
            result = list(
                iterate_in_subinterpreter(
                    _Wrap(ref, pre=_check_flag0and1),
                    initializer=[_init_flag0, _init_flag1],
                    timeout=5.0,
                )
            )
            self.assertEqual(result, ref)
            # The flag should not be set in the main interpreter
            self.assertEqual(_FLAGS, [])

        def test_partial_iteration(self) -> None:
            """Test partial iteration by breaking early."""
            iterable = iterate_in_subinterpreter(_Wrap(range(10)))
            result = []
            for i, item in enumerate(iterable):
                result.append(item)
                if i >= 4:
                    break
            self.assertEqual(result, [0, 1, 2, 3, 4])


# Module-level functions and classes (required for pickling/subinterpreter compatibility)
def _double(x: int) -> int:
    """Helper function to double a value."""
    return x * 2


def _only_even(x: int) -> int | None:
    """Helper function to filter only even numbers."""
    return x if x % 2 == 0 else None


class _StatefulSource:
    """Stateful source that tracks iteration calls."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.calls = 0

    def __iter__(self) -> Iterator[int]:
        start = self.calls * self.n
        self.calls += 1
        yield from range(start, start + self.n)


class _validate_pipeline_id:
    """Helper class to validate that the pipeline ID is as expected."""

    def __init__(self, val: int) -> None:
        self.val = val

    def __iter__(self) -> Iterator[int]:
        if (v := _get_global_id()) != self.val:
            raise AssertionError(f"_node._PIPELINE_ID={v} != {self.val=}")
        yield 0


if sys.version_info >= (3, 14):

    class TestRunPipelineInSubinterpreter(unittest.TestCase):
        """Test cases for run_pipeline_in_subinterpreter function."""

        def test_basic_pipeline(self) -> None:
            """Test basic pipeline execution in subinterpreter."""
            condig = PipelineBuilder().add_source(range(5)).add_sink().get_config()
            iterable = run_pipeline_in_subinterpreter(condig, num_threads=1, timeout=5)
            result = list(iterable)
            self.assertEqual(result, [0, 1, 2, 3, 4])

        def test_pipeline_with_pipe(self) -> None:
            """Test pipeline with pipe operation in subinterpreter."""
            config = (
                PipelineBuilder()
                .add_source(range(5))
                .pipe(_double)
                .add_sink()
                .get_config()
            )
            iterable = run_pipeline_in_subinterpreter(config, num_threads=1, timeout=5)
            result = list(iterable)
            self.assertEqual(result, [0, 2, 4, 6, 8])

        def test_pipeline_multiple_iterations(self) -> None:
            """Test that the pipeline can be iterated multiple times."""
            config = (
                PipelineBuilder()
                .add_source(_StatefulSource(3))
                .add_sink(buffer_size=10)
                .get_config()
            )
            iterable = run_pipeline_in_subinterpreter(config, num_threads=1, timeout=5)

            # First iteration
            result1 = list(iterable)
            self.assertEqual(result1, [0, 1, 2])

            # Second iteration should start from where we left off
            result2 = list(iterable)
            self.assertEqual(result2, [3, 4, 5])

        def test_pipeline_with_aggregate(self) -> None:
            """Test pipeline with aggregation in subinterpreter."""
            config = (
                PipelineBuilder()
                .add_source(range(10))
                .aggregate(3)
                .add_sink(buffer_size=10)
                .get_config()
            )
            iterable = run_pipeline_in_subinterpreter(config, num_threads=1, timeout=5)
            result = list(iterable)
            self.assertEqual(result, [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]])

        def test_pipeline_empty_source(self) -> None:
            """Test pipeline with empty source in subinterpreter."""
            config = PipelineBuilder().add_source([]).add_sink().get_config()
            iterable = run_pipeline_in_subinterpreter(config, num_threads=1, timeout=5)
            result = list(iterable)
            self.assertEqual(result, [])

        def test_pipeline_with_filter(self) -> None:
            """Test pipeline with filter operation (returning None to skip items)."""
            config = (
                PipelineBuilder()
                .add_source(range(10))
                .pipe(_only_even)
                .add_sink()
                .get_config()
            )
            iterable = run_pipeline_in_subinterpreter(config, num_threads=1, timeout=5)
            result = list(iterable)
            self.assertEqual(result, [0, 2, 4, 6, 8])

        def test_pipeline_with_buffer_size(self) -> None:
            """Test run_pipeline_in_subinterpreter with custom buffer_size."""
            config = PipelineBuilder().add_source(range(10)).add_sink().get_config()
            iterable = run_pipeline_in_subinterpreter(
                config, num_threads=1, buffer_size=5, timeout=5
            )
            result = list(iterable)
            self.assertEqual(result, list(range(10)))

        def test_run_pipeline_in_subinterpreter_pipeline_id(self) -> None:
            """Test pipeline inherits global ID in subinterpreter."""
            # Set to a number that's not zero and something unlikely to
            # happen during testing
            _set_global_id(123456)
            ref = _get_global_id() + 1

            config = (
                PipelineBuilder()
                .add_source(_validate_pipeline_id(ref))
                .add_sink()
                .get_config()
            )

            iterable = run_pipeline_in_subinterpreter(config, num_threads=1, timeout=5)

            for _ in iterable:
                pass

else:

    class TestRunPipelineInSubinterpreter(unittest.TestCase):
        """Placeholder tests for Python < 3.14."""

        def test_requires_python_3_14(self) -> None:
            """Test that run_pipeline_in_subinterpreter requires Python 3.14+."""
            config = PipelineBuilder().add_source([1, 2, 3]).add_sink().get_config()
            with self.assertRaises(RuntimeError) as cm:
                run_pipeline_in_subinterpreter(config, num_threads=1)
            self.assertIn("Python 3.14", str(cm.exception))
