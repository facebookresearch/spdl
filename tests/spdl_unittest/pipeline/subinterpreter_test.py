# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import sys
import unittest
from collections.abc import Callable, Iterable, Sequence
from typing import Generic, TypeVar

from parameterized import parameterized
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
