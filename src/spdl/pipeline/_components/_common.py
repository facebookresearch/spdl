# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["_queue_stage_hook", "_EOF"]

from contextlib import asynccontextmanager
from typing import AsyncGenerator, TypeVar

from .._queue import AsyncQueue

# pyre-strict


T = TypeVar("T")
U = TypeVar("U")


# Sentinel objects used to instruct AsyncPipeline to take special actions.
class _Sentinel:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name


_EOF = _Sentinel("EOF")  # Indicate the end of stream.


@asynccontextmanager
async def _queue_stage_hook(queue: AsyncQueue[T]) -> AsyncGenerator[None, None]:
    # Responsibility
    #   1. Call the `stage_hook`` context manager
    #   2. Put _EOF when the stage is done for reasons other than cancel.

    # Note:
    # `asyncio.CancelledError` is a subclass of BaseException, so it won't be
    # caught in the following, and EOF won't be passed to the output queue.
    async with queue.stage_hook():
        try:
            yield
        except Exception:
            await queue.put(_EOF)  # pyre-ignore: [6]
            raise
        else:
            await queue.put(_EOF)  # pyre-ignore: [6]
