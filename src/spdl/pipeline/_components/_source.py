# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["_source", "_source_continuous"]

import asyncio
import sys
import threading
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Iterable
from contextlib import aclosing
from typing import cast, Generic, TypeVar

from ._common import _drive_to_completion, _EPOCH_END
from ._queue import _queue_stage_hook, AsyncQueue

T = TypeVar("T")

_DONE = object()
_LOOP_STATE_POLL_INTERVAL = 0.1


class _SourceError:
    def __init__(self, error: BaseException) -> None:
        self.error = error


class _SyncSourceThread(Generic[T]):
    def __init__(self, source: Iterable[T], *, continuous: bool) -> None:
        self._source = source
        self._continuous = continuous
        self._loop = asyncio.get_running_loop()
        self._messages: asyncio.Queue[object] = asyncio.Queue()
        self._acknowledged = threading.Event()
        self._stopped = threading.Event()
        self._source_call_lock = threading.Lock()
        self._thread_started = False
        self._thread = threading.Thread(
            target=self._run,
            name="spdl_sync_source",
            daemon=False,
        )

    def _run(self) -> None:
        try:
            while True:
                if not self._iterate_once():
                    return
                if not self._continuous:
                    break
                if not self._send(_EPOCH_END):
                    return
        except (Exception, KeyboardInterrupt, SystemExit) as error:
            self._send(_SourceError(error))
        else:
            self._send(_DONE)

    def _iterate_once(self) -> bool:
        with self._source_call_lock:
            if self._stopped.is_set():
                return False
            iterator = iter(self._source)

        exhausted = False
        try:
            while True:
                with self._source_call_lock:
                    if self._stopped.is_set():
                        return False
                    try:
                        item = next(iterator)
                    except StopIteration:
                        exhausted = True
                        return True
                if not self._send(item):
                    return False
        finally:
            if (
                not exhausted
                and (close := getattr(iterator, "close", None)) is not None
            ):
                source_error = sys.exc_info()[1]
                try:
                    close()
                except BaseException:
                    if source_error is None:
                        raise

    def _send(self, message: object) -> bool:
        self._acknowledged.clear()
        if self._stopped.is_set():
            return False
        try:
            self._loop.call_soon_threadsafe(self._publish, message)
        except RuntimeError:
            self._stopped.set()
            return False
        while not self._acknowledged.wait(_LOOP_STATE_POLL_INTERVAL):
            if not self._loop.is_running():
                self._stopped.set()
                return False
        return not self._stopped.is_set()

    def _publish(self, message: object) -> None:
        if self._stopped.is_set():
            self._acknowledged.set()
            return
        self._messages.put_nowait(message)

    def _acknowledge(self) -> None:
        self._acknowledged.set()

    def _stop(self) -> None:
        self._stopped.set()
        self._acknowledged.set()
        if self._thread_started and self._source_call_lock.acquire(blocking=False):
            self._source_call_lock.release()
            self._thread.join()

    async def iterate(self) -> AsyncGenerator[T | object, None]:
        try:
            self._thread.start()
            self._thread_started = True
            while True:
                message = await self._messages.get()
                if message is _DONE:
                    self._acknowledge()
                    return
                if isinstance(message, _SourceError):
                    self._acknowledge()
                    raise message.error
                try:
                    yield cast(T, message)
                except BaseException:
                    self._stop()
                    raise
                else:
                    self._acknowledge()
        finally:
            self._stop()


async def _close_async_source(source: object) -> None:
    if (close := getattr(source, "aclose", None)) is None:
        return
    source_error = sys.exc_info()[1]
    try:
        _, cancelled = await _drive_to_completion(close())
    except BaseException:
        if source_error is None:
            raise
    else:
        if cancelled is not None and source_error is None:
            raise cancelled


async def _source(
    src: Iterable[T] | AsyncIterable[T],
    queue: AsyncQueue,
    max_items: int | None = None,
) -> None:
    """Coroutine that generates data from an iterator and puts it into a queue.

    This coroutine represents the source stage of a pipeline. It consumes items from
    a synchronous or asynchronous iterable and puts them into the output queue for
    downstream processing. The coroutine completes when the iterator is exhausted or
    when the maximum number of items has been reached.

    Args:
        src: The source iterable (synchronous or asynchronous) to consume data from.
        queue: The async queue to put items into for downstream consumption.
        max_items: Optional maximum number of items to process. If ``None``, processes
            all items from the source.

    Note:
        The EOF token is automatically handled by the queue's stage hook context manager,
        so this coroutine does not explicitly send EOF.
    """
    async with _queue_stage_hook(queue):
        num_items = 0
        if hasattr(src, "__aiter__"):
            items = aiter(cast(AsyncIterable[T], src))
            try:
                async for item in items:
                    await queue.put(item)
                    num_items += 1
                    if max_items is not None and num_items >= max_items:
                        return
            finally:
                await _close_async_source(items)
            return

        runner = _SyncSourceThread(cast(Iterable[T], src), continuous=False)
        async with aclosing(runner.iterate()) as items:
            async for item in items:
                await queue.put(item)
                num_items += 1
                if max_items is not None and num_items >= max_items:
                    return


async def _source_continuous(
    src: Iterable[T] | AsyncIterable[T],
    queue: AsyncQueue,
) -> None:
    """Coroutine that continuously re-iterates the source, injecting epoch-end markers.

    After each iteration of the source exhausts, an ``_EPOCH_END`` sentinel is
    put into the queue and the source is re-iterated. This continues indefinitely
    until the coroutine is cancelled (via pipeline shutdown).

    Args:
        src: The source iterable (synchronous or asynchronous) to consume data from.
        queue: The async queue to put items into for downstream consumption.
    """
    async with _queue_stage_hook(queue):
        if hasattr(src, "__aiter__"):
            async_items: AsyncIterator[T] | None = None
            try:
                while True:
                    async_items = aiter(cast(AsyncIterable[T], src))
                    try:
                        async for item in async_items:
                            await queue.put(item)
                    finally:
                        if async_items is not src:
                            await _close_async_source(async_items)
                    await queue.put(_EPOCH_END)
            finally:
                if async_items is src:
                    await _close_async_source(async_items)

        runner = _SyncSourceThread(cast(Iterable[T], src), continuous=True)
        async with aclosing(runner.iterate()) as items:
            async for item in items:
                await queue.put(item)
