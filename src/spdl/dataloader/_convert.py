# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import inspect
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


def _to_async(
    func: Callable[[T], U],
    executor: type[Executor] | None,
) -> Callable[[T], Awaitable[U]]:
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
        for result in await loop.run_in_executor(executor, _wrap_gen, func, item):
            yield result

    return afunc


def _to_async_gen(
    func: Callable[[T], Iterable[U]],
    executor: type[Executor] | None,
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


def validate_op(
    op: Callables[T, U],
    executor: type[Executor] | None,
    output_order: str,
) -> None:
    if inspect.iscoroutinefunction(op) or inspect.isasyncgenfunction(op):
        if executor is not None:
            raise ValueError("`executor` cannot be specified when op is async.")
    if inspect.isasyncgenfunction(op):
        if output_order == "input":
            raise ValueError(
                "pipe does not support async generator function "
                "when output_order is 'input'."
            )


def convert_to_async(
    op: Callables[T, U],
    executor: type[Executor] | None,
) -> AsyncCallables[T, U]:
    if inspect.iscoroutinefunction(op) or inspect.isasyncgenfunction(op):
        # op is async function. No need to convert.
        return op  # pyre-ignore: [7]

    if inspect.isgeneratorfunction(op):
        # op is sync generator. Convert to async generator.
        if isinstance(executor, ProcessPoolExecutor):
            # If executing in subprocess, data can be only exchanged at the end
            # of the generator.
            return _to_batch_async_gen(op, executor=executor)
        return _to_async_gen(op, executor=executor)

    # Convert a regular sync function to async function.
    return _to_async(op, executor=executor)
