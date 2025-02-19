# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TypeVar

from ._hook import _time_str, StatsCounter

__all__ = [
    "AsyncQueue",
    "StatsQueue",
]

T = TypeVar("T")
_LG: logging.Logger = logging.getLogger(__name__)


class AsyncQueue(asyncio.Queue[T]):
    """Extends :py:class:`asyncio.Queue` with init/finalize logic.

    When a pipeline stage starts, :py:class:`PipelineBuilder` calls
    :py:meth:`~AsyncQueue.stage_hook` method, and the initialization logic
    is executed. During the execution of the pipeline, ``get``/``put`` methods
    are used to pass data. At the end of pipeline execution, the finalization
    logic in :py:meth:`~AsyncQueue.stage_hook` is executed.


    One intended usage is to overload the ``get``/``put`` methods to capture the
    time each pipeline stage waits, then publish the stats. This helps identifying
    the bottleneck in the pipeline. See :py:class:`StatsQueue`.

    Args:
        name: The name of the queue. Assigned by :py:class:`PipelineBuilder`.
        buffer_size: The buffer size. Assigned by :py:class:`PipelineBuilder`.
    """

    # The default value is for simplifying unit testing.
    def __init__(self, name: str, buffer_size: int = 0) -> None:
        super().__init__(buffer_size)
        self.name = name

    @asynccontextmanager
    async def stage_hook(self) -> AsyncIterator[None]:
        """Context manager, which handles init/final logic for the stage."""
        yield


class StatsQueue(AsyncQueue[T]):
    """Measures the time stages are blocked on upstream/downstream stage.
    Extends :py:class:`AsyncQueue`.

    Args:
        name: The name of the queue. Assigned by :py:class:`PipelineBuilder`.
        buffer_size: The buffer size. Assigned by :py:class:`PipelineBuilder`.
    """

    def __init__(self, name: str, buffer_size: int = 0) -> None:
        super().__init__(name, buffer_size)

        self._getc = StatsCounter()
        self._putc = StatsCounter()

    async def get(self) -> T:
        """Remove and return an item from the queue, track the time."""
        with self._getc.count():
            return await super().get()

    async def put(self, item: T) -> None:
        """Remove and return an item from the queue, track the time."""
        with self._putc.count():
            return await super().put(item)

    @asynccontextmanager
    async def stage_hook(self) -> AsyncIterator[None]:
        t0 = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - t0
            self._log_stats(
                self._getc.num_items,
                elapsed,
                self._putc.ave_time,
                self._getc.ave_time,
            )

    def _log_stats(
        self, num_items: int, elapsed: float, put_time: float, get_time: float
    ) -> None:
        qps = num_items / elapsed
        _LG.info(
            "[%s]\tProcessed %5d items in %s (QPS: %6.1f) "
            "Ave wait time: Upstream (put): %s, Downstream (get): %s.",
            self.name,
            num_items,
            _time_str(elapsed),
            qps,
            _time_str(put_time),
            _time_str(get_time),
        )
