# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Unit tests for the source_locator module.
"""

import asyncio
import functools
import inspect
import unittest
from typing import Any

from spdl.pipeline._common._source_locator import locate_source


def regular_function(x: int, y: int) -> int:
    """A regular function for testing."""
    return x + y


def generator_function(n: int) -> Any:
    """A generator function for testing."""
    for i in range(n):
        yield i


async def async_function(x: int) -> int:
    """An async function for testing."""
    await asyncio.sleep(0)
    return x * 2


async def async_generator_function(n: int) -> Any:
    """An async generator function for testing."""
    for i in range(n):
        await asyncio.sleep(0)
        yield i


class SimpleCallable:
    """A simple callable class for testing."""

    def __call__(self, x: int) -> int:
        return x * 3


def _ln(target: object) -> int:
    return inspect.getsourcelines(target)[1]  # pyre-ignore[6]


class TestSourceLocator(unittest.TestCase):
    """Test cases for the locate_source function."""

    def test_regular_function(self) -> None:
        """Test locating source for a regular function."""
        loc = locate_source(regular_function)

        self.assertEqual(loc.name, f"{__name__}.regular_function")
        self.assertEqual(loc.file_path, __file__)
        self.assertEqual(loc.line_number, _ln(regular_function))
        self.assertEqual(loc.partial_args, ())
        self.assertEqual(loc.partial_kwargs, {})

    def test_generator_function(self) -> None:
        """Test locating source for a generator function."""
        loc = locate_source(generator_function)

        self.assertEqual(loc.name, f"{__name__}.generator_function")
        self.assertEqual(loc.file_path, __file__)
        self.assertEqual(loc.line_number, _ln(generator_function))
        self.assertEqual(loc.partial_args, ())
        self.assertEqual(loc.partial_kwargs, {})

    def test_async_function(self) -> None:
        """Test locating source for an async function."""
        loc = locate_source(async_function)

        self.assertEqual(loc.name, f"{__name__}.async_function")
        self.assertEqual(loc.file_path, __file__)
        self.assertEqual(loc.line_number, _ln(async_function))
        self.assertEqual(loc.partial_args, ())
        self.assertEqual(loc.partial_kwargs, {})

    def test_async_generator_function(self) -> None:
        """Test locating source for an async generator function."""
        loc = locate_source(async_generator_function)

        self.assertEqual(loc.name, f"{__name__}.async_generator_function")
        self.assertEqual(loc.file_path, __file__)
        self.assertEqual(loc.line_number, _ln(async_generator_function))
        self.assertEqual(loc.partial_args, ())
        self.assertEqual(loc.partial_kwargs, {})

    def test_callable_class_object(self) -> None:
        """Test locating source for a callable class object."""
        obj = SimpleCallable()
        loc = locate_source(obj)

        self.assertEqual(loc.name, f"{__name__}.SimpleCallable")
        self.assertEqual(loc.file_path, __file__)
        self.assertEqual(loc.line_number, _ln(SimpleCallable))
        self.assertEqual(loc.partial_args, ())
        self.assertEqual(loc.partial_kwargs, {})

    def test_builtin_function(self) -> None:
        """Test locating source for a built-in function."""
        loc = locate_source(len)

        self.assertEqual(loc.name, "builtins.len")
        self.assertIsNone(loc.file_path)
        self.assertIsNone(loc.line_number)
        self.assertEqual(loc.partial_args, ())
        self.assertEqual(loc.partial_kwargs, {})

    def test_partial_with_positional_args(self) -> None:
        """
        Test locating source for a function wrapped with functools.partial
        (positional args).
        """
        partial_func = functools.partial(regular_function, 5)
        loc = locate_source(partial_func)

        self.assertEqual(loc.name, f"{__name__}.regular_function")
        self.assertEqual(loc.file_path, __file__)
        self.assertIsNotNone(loc.line_number)
        self.assertGreater(loc.line_number, 0)
        self.assertEqual(loc.partial_args, (5,))
        self.assertEqual(loc.partial_kwargs, {})

    def test_partial_with_keyword_args(self) -> None:
        """
        Test locating source for a function wrapped with functools.partial
        (keyword args).
        """
        partial_func = functools.partial(regular_function, y=10)
        loc = locate_source(partial_func)

        self.assertEqual(loc.name, f"{__name__}.regular_function")
        self.assertEqual(loc.file_path, __file__)
        self.assertIsNotNone(loc.line_number)
        self.assertGreater(loc.line_number, 0)
        self.assertEqual(loc.partial_args, ())
        self.assertEqual(loc.partial_kwargs, {"y": 10})

    def test_callable_object_wrapped_with_partial(self) -> None:
        """
        Test locating source for a callable object wrapped with
        functools.partial.
        """
        obj = SimpleCallable()
        partial_obj = functools.partial(obj, 7)
        loc = locate_source(partial_obj)

        self.assertEqual(loc.name, f"{__name__}.SimpleCallable")
        self.assertEqual(loc.file_path, __file__)
        self.assertIsNotNone(loc.line_number)
        self.assertGreater(loc.line_number, 0)
        self.assertEqual(loc.partial_args, (7,))
        self.assertEqual(loc.partial_kwargs, {})

    def test_nested_partial(self) -> None:
        """Test locating source for nested functools.partial wrapping."""
        partial_func1 = functools.partial(regular_function, 3)
        partial_func2 = functools.partial(partial_func1, y=8)
        loc = locate_source(partial_func2)

        self.assertEqual(loc.name, f"{__name__}.regular_function")
        self.assertEqual(loc.file_path, __file__)
        self.assertIsNotNone(loc.line_number)
        self.assertGreater(loc.line_number, 0)
        self.assertEqual(loc.partial_args, (3,))
        self.assertEqual(loc.partial_kwargs, {"y": 8})

    def test_nested_partial_positional_args_order(self) -> None:
        """Test that nested partial positional args are in correct order."""
        # partial(partial(f, 1), 2) should produce args (1, 2)
        partial_func1 = functools.partial(regular_function, 1)
        partial_func2 = functools.partial(partial_func1, 2)
        loc = locate_source(partial_func2)

        self.assertEqual(loc.partial_args, (1, 2))
        # Verify the actual behavior matches
        self.assertEqual(partial_func2(), regular_function(1, 2))

    def test_nested_partial_keyword_override(self) -> None:
        """Test that outer partial keywords override inner ones."""
        # partial(partial(f, x=1), x=2) should use x=2
        partial_func1 = functools.partial(regular_function, x=1, y=5)
        partial_func2 = functools.partial(partial_func1, x=2)
        loc = locate_source(partial_func2)

        self.assertEqual(loc.partial_kwargs, {"x": 2, "y": 5})
        # Verify the actual behavior matches
        self.assertEqual(partial_func2(), regular_function(x=2, y=5))
