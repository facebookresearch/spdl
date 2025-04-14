# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import inspect
import sys
import traceback
from collections.abc import AsyncIterable, Awaitable, Callable, Iterable, Iterator
from concurrent.futures import Executor, ProcessPoolExecutor
from typing import TypeAlias, TypeVar

T = TypeVar("T")
U = TypeVar("U")


Callables: TypeAlias = (
    Callable[[T], U]
    | Callable[[T], Iterable[U]]
    | Callable[[T], Awaitable[U]]
    | Callable[[T], AsyncIterable[U]]
)
AsyncCallables: TypeAlias = (
    Callable[[T], Awaitable[U]] | Callable[[T], AsyncIterable[U]]
)


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
    if isinstance(executor, ProcessPoolExecutor):

        async def afunc(item: T) -> U:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(executor, _func_in_subprocess, func, item)

    else:

        async def afunc(item: T) -> U:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(executor, func, item)

    return afunc


def _wrap_gen(generator: Callable[[T], Iterable[U]], item: T) -> list[U]:
    return list(generator(item))


def _to_batch_async_gen(
    func: Callable[[T], Iterable[U]],
    executor: ProcessPoolExecutor,
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
                return next(gen)
            except StopIteration:
                return sentinel  # type: ignore[return-value]

        while (val := await loop.run_in_executor(executor, _next)) is not sentinel:
            yield val

    return afunc


def convert_to_async(
    op: Callables[T, U],
    executor: Executor | None,
) -> AsyncCallables[T, U]:
    if inspect.iscoroutinefunction(op) or inspect.isasyncgenfunction(op):
        # op is async function. No need to convert.
        assert executor is None  # This has been checked in `PipelineBuilder.pipe()`
        return op  # pyre-ignore: [7]

    if inspect.isgeneratorfunction(op):
        # op is sync generator. Convert to async generator.
        if isinstance(executor, ProcessPoolExecutor):
            # If executing in subprocess, data can be only exchanged at the end
            # of the generator.
            return _to_batch_async_gen(op, executor=executor)
        return _to_async_gen(op, executor=executor)

    # Convert a regular sync function to async function.
    return _to_async(op, executor=executor)  # pyre-ignore: [7]
