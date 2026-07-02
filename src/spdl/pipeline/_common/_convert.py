# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

__all__ = [
    "convert_to_async",
    "_is_process_pool",
    "_is_interpreter_pool",
    "_is_isolating_pool",
    "_to_async_gen",
]

import asyncio
import inspect
import sys
import traceback
from collections.abc import AsyncIterable, Awaitable, Callable, Iterable, Iterator
from concurrent.futures import Executor, ProcessPoolExecutor
from typing import TypeVar

from ._types import _TAsyncCallables, _TCallables

T = TypeVar("T")
U = TypeVar("U")


def _is_process_pool(executor: Executor | type[Executor] | None) -> bool:
    """Check if executor is or wraps a ProcessPoolExecutor."""
    if isinstance(executor, ProcessPoolExecutor):
        return True
    pool_cls = getattr(executor, "_pool_executor_class", None)
    if pool_cls is not None:
        return issubclass(pool_cls, ProcessPoolExecutor)
    # PriorityExecutorEntrypoint: resolve via _owner property
    owner = getattr(executor, "_owner", None)
    if owner is not None:
        return _is_process_pool(owner)
    return False


# Declared before the branch so its type stays ``type[Executor] | None`` on every Python
# version; otherwise, on versions without ``InterpreterPoolExecutor`` the ``else`` assignment
# narrows it to ``None`` and the isinstance/issubclass checks below fail to type-check.
_INTERPRETER_POOL_CLASS: type[Executor] | None
if sys.version_info >= (3, 14):
    from concurrent.futures.interpreter import (  # pyre-ignore[21]
        InterpreterPoolExecutor,
    )

    _INTERPRETER_POOL_CLASS = InterpreterPoolExecutor
else:
    _INTERPRETER_POOL_CLASS = None


def _is_interpreter_pool(executor: Executor | type[Executor] | None) -> bool:
    """Check whether ``executor`` is or wraps an ``InterpreterPoolExecutor``.

    Mirrors :py:func:`_is_process_pool`: matches the stdlib ``InterpreterPoolExecutor``
    (Python 3.14+) directly, SPDL pool wrappers via their ``_pool_executor_class``, and a
    ``PriorityExecutorEntrypoint`` via its ``_owner``.

    Args:
        executor: The executor (or executor class, or ``None``) to inspect.

    Returns:
        ``True`` if the executor isolates work in a subinterpreter, else ``False``. Always
        ``False`` on Python versions without ``InterpreterPoolExecutor``.
    """
    if _INTERPRETER_POOL_CLASS is None or executor is None:
        return False
    if isinstance(executor, _INTERPRETER_POOL_CLASS):
        return True
    pool_cls = getattr(executor, "_pool_executor_class", None)
    if pool_cls is not None:
        return issubclass(pool_cls, _INTERPRETER_POOL_CLASS)
    owner = getattr(executor, "_owner", None)
    if owner is not None:
        return _is_interpreter_pool(owner)
    return False


def _is_isolating_pool(executor: Executor | type[Executor] | None) -> bool:
    """Check whether ``executor`` runs work in a separate process or subinterpreter.

    These are exactly the executors whose inter-stage handoff crosses an IPC / pickling
    boundary, and therefore the executors that fusion targets. Thread pools (which share
    address space) and ``None`` (the default thread pool) are not isolating.

    Args:
        executor: The executor (or executor class, or ``None``) to inspect.

    Returns:
        ``True`` if the executor is a process pool or interpreter pool, else ``False``.
    """
    return _is_process_pool(executor) or _is_interpreter_pool(executor)


def _func_in_subprocess(func: Callable[[T], U], item: T) -> U:
    try:
        return func(item)
    except Exception as err:
        # When running a function in subprocess, the exception context is lost.
        # This makes it difficult to debug, so we add extra layer to pass
        # the source code location back to the main process
        _, _, exc_tb = sys.exc_info()
        f = traceback.extract_tb(exc_tb, limit=-1)[-1]

        raise RuntimeError(
            "Function failed in subprocess: "
            f"{type(err).__name__}: {err} ({f.filename}:{f.lineno}:{f.name})"
        ) from None


def _to_async(
    func: Callable[[T], U],
    executor: type[Executor] | None,
) -> Callable[[T], Awaitable[U]]:
    if _is_process_pool(executor):

        async def afunc(item: T) -> U:
            loop = asyncio.get_running_loop()
            # pyrefly: ignore [bad-argument-type]
            return await loop.run_in_executor(executor, _func_in_subprocess, func, item)

    else:

        async def afunc(item: T) -> U:
            loop = asyncio.get_running_loop()
            # pyrefly: ignore [bad-argument-type]
            return await loop.run_in_executor(executor, func, item)

    return afunc


def _wrap_gen(generator: Callable[[T], Iterable[U]], item: T) -> list[U]:
    return list(generator(item))


def _to_batch_async_gen(
    func: Callable[[T], Iterable[U]],
    executor: Executor,
) -> Callable[[T], AsyncIterable[U]]:
    async def afunc(item: T) -> AsyncIterable[U]:
        loop = asyncio.get_running_loop()
        # pyre-ignore: [6]
        for result in await loop.run_in_executor(executor, _wrap_gen, func, item):
            yield result

    return afunc


def _to_async_gen(
    func: Callable[[T], Iterable[U]],
    executor: Executor | None,
) -> Callable[[T], AsyncIterable[U]]:
    async def afunc(item: T) -> AsyncIterable[U]:
        loop = asyncio.get_running_loop()
        gen: Iterator[U] = iter(await loop.run_in_executor(executor, func, item))
        # Note on the use of sentinel
        #
        # One would think that catching StopIteration is simpler here, like
        #
        # while True:
        #     try:
        #         yield await run_async(next, gen)
        #     except StopIteration:
        #         break
        #
        # Unfortunately, this does not work. It throws an error like
        # `TypeError: StopIteration interacts badly with generators and
        # cannot be raised into a Future`
        #
        # To workaround, we handle StopIteration in the sync function, and notify
        # the end of Iteratoin with sentinel object.
        sentinel: object = object()

        def _next() -> U:
            """Wrapper around generator.
            This is necessary as we cannot raise StopIteration in async func."""
            try:
                item = next(gen)
            except StopIteration:
                return sentinel  # type: ignore[return-value]
            else:
                return item

        while (val := await loop.run_in_executor(executor, _next)) is not sentinel:
            yield val

    return afunc


def convert_to_async(
    op: _TCallables[T, U],
    executor: Executor | None,
) -> _TAsyncCallables[T, U]:
    if inspect.ismethod(op.__call__):
        op = op.__call__

    if inspect.iscoroutinefunction(op) or inspect.isasyncgenfunction(op):
        # op is async function. No need to convert.
        assert executor is None  # This has been checked in `PipelineBuilder.pipe()`
        return op  # pyre-ignore: [7]

    if inspect.isgeneratorfunction(op):
        # op is sync generator. Convert to async generator.
        if _is_process_pool(executor):
            # If executing in subprocess, data can be only exchanged at the end
            # of the generator.
            assert isinstance(executor, Executor)
            return _to_batch_async_gen(op, executor=executor)
        return _to_async_gen(op, executor=executor)

    # Convert a regular sync function to async function.
    return _to_async(op, executor=executor)  # pyre-ignore: [7]
