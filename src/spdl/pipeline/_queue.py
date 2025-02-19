# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TypeVar

__all__ = [
    "AsyncQueue",
]

T = TypeVar("T")


class AsyncQueue(asyncio.Queue[T]):
    # The default value is for simplifying unit testing.
    def __init__(self, name: str, buffer_size: int = 0) -> None:
        super().__init__(buffer_size)
        self.name = name

    @asynccontextmanager
    async def stage_hook(self) -> AsyncIterator[None]:
        yield
