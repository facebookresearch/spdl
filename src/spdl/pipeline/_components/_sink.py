# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["_sink"]

from typing import TypeVar

from .._queue import AsyncQueue
from ._common import _EOF

# pyre-strict


T = TypeVar("T")
U = TypeVar("U")


async def _sink(input_queue: AsyncQueue[T], output_queue: AsyncQueue[T]) -> None:
    async with output_queue.stage_hook():
        while True:
            item = await input_queue.get()

            if item is _EOF:
                break
            await output_queue.put(item)
