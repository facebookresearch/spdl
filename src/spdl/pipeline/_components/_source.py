# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["_source"]

from collections.abc import AsyncIterable, Iterable
from typing import TypeVar

from spdl.pipeline._common._convert import _to_async_gen

from ._queue import _queue_stage_hook, AsyncQueue

# pyre-strict

T = TypeVar("T")
U = TypeVar("U")


async def _source(
    src: Iterable[T] | AsyncIterable[T],
    queue: AsyncQueue,
    max_items: int | None = None,
) -> None:
    src_: AsyncIterable[T] = (  # pyre-ignore: [9]
        src if hasattr(src, "__aiter__") else _to_async_gen(iter, None)(src)
    )

    async with _queue_stage_hook(queue):
        num_items = 0
        async for item in src_:
            await queue.put(item)
            num_items += 1
            if max_items is not None and num_items >= max_items:
                return
