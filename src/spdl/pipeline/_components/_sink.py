# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["_sink"]

from typing import TypeVar

from ._common import is_eof
from ._queue import AsyncQueue

# pyre-strict


T = TypeVar("T")
U = TypeVar("U")


async def _sink(input_queue: AsyncQueue, output_queue: AsyncQueue) -> None:
    """Coroutine that consumes data from input queue and buffers it in output queue.

    This coroutine represents the sink stage of a pipeline. It consumes items from
    the input queue and puts them into the output queue for the foreground thread to
    fetch. The coroutine completes when it receives an EOF token, which it does not
    pass to the output queue (EOF filtering is the sink's responsibility).

    Args:
        input_queue: The async queue to consume items from (typically from the last
            processing stage).
        output_queue: The async queue to buffer items for the foreground thread.

    Note:
        The sink filters out EOF tokens and does not pass them to the output queue.
        The EOF handling and queue cleanup is managed by the queue's stage hook
        context manager.
    """
    async with output_queue.stage_hook():
        while True:
            item = await input_queue.get()

            if is_eof(item):
                break
            await output_queue.put(item)
