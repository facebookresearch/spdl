# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import inspect
import unittest
from collections.abc import AsyncIterator, Iterable
from functools import partial

from spdl.pipeline._common._convert import (
    _is_async_callable,
    _is_asyncgen_callable,
    _is_callable,
    _is_coroutine_callable,
)


def _sync_function(value: object) -> object:
    return value


async def _coroutine_function(value: object) -> object:
    return value


async def _coroutine_function_with_prefix(
    prefix: str,
    value: object,
) -> tuple[str, object]:
    return prefix, value


async def _asyncgen_function(values: Iterable[object]) -> AsyncIterator[object]:
    for value in values:
        yield value


class _SyncCallable:
    def __call__(self, value: object) -> object:
        return value


class _CoroutineCallable:
    async def __call__(self, value: object) -> object:
        return value


class _AsyncGeneratorCallable:
    async def __call__(self, values: Iterable[object]) -> AsyncIterator[object]:
        for value in values:
            yield value


class CallableDetectionTest(unittest.TestCase):
    def test_is_callable_accepts_functions_and_callable_instances(self) -> None:
        """Callable detection accepts functions and objects defining __call__."""
        self.assertTrue(_is_callable(_sync_function))
        self.assertTrue(_is_callable(_SyncCallable()))
        self.assertTrue(_is_callable(_CoroutineCallable()))
        self.assertFalse(_is_callable(object()))

    def test_is_coroutine_callable_inspects_instance_call_method(self) -> None:
        """Coroutine detection recognizes an instance with async __call__."""
        op = _CoroutineCallable()

        self.assertTrue(callable(op))
        self.assertFalse(inspect.iscoroutinefunction(op))
        self.assertTrue(inspect.iscoroutinefunction(op.__call__))
        self.assertTrue(_is_coroutine_callable(op))

    def test_is_coroutine_callable_accepts_functions_and_partials(self) -> None:
        """Coroutine detection accepts async functions and their partials."""
        self.assertTrue(_is_coroutine_callable(_coroutine_function))
        self.assertTrue(
            _is_coroutine_callable(partial(_coroutine_function_with_prefix, "prefix"))
        )
        self.assertFalse(_is_coroutine_callable(_sync_function))
        self.assertFalse(_is_coroutine_callable(_SyncCallable()))

    def test_is_asyncgen_callable_inspects_instance_call_method(self) -> None:
        """Async-generator detection recognizes functions and callable instances."""
        op = _AsyncGeneratorCallable()

        self.assertFalse(inspect.isasyncgenfunction(op))
        self.assertTrue(inspect.isasyncgenfunction(op.__call__))
        self.assertTrue(_is_asyncgen_callable(op))
        self.assertTrue(_is_asyncgen_callable(_asyncgen_function))
        self.assertFalse(_is_asyncgen_callable(_coroutine_function))

    def test_is_async_callable_combines_async_callable_kinds(self) -> None:
        """Combined detection accepts coroutine and async-generator callables."""
        self.assertTrue(_is_async_callable(_coroutine_function))
        self.assertTrue(_is_async_callable(_CoroutineCallable()))
        self.assertTrue(_is_async_callable(_asyncgen_function))
        self.assertTrue(_is_async_callable(_AsyncGeneratorCallable()))
        self.assertFalse(_is_async_callable(_sync_function))
        self.assertFalse(_is_async_callable(_SyncCallable()))
