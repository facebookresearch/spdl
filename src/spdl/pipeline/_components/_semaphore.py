# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Resizable asyncio semaphore for dynamic concurrency control."""

__all__ = ["ResizableSemaphore"]

import asyncio
from collections import deque


class ResizableSemaphore:
    """asyncio.Semaphore variant whose max value can be changed at runtime.

    Thread safety: asyncio is single-threaded per event loop. All methods
    MUST be called from coroutines in the same event loop. No locks needed.

    Resize semantics:
      - Increase: immediately wake up to ``new_max - old_max`` blocked
        waiters.
      - Decrease: no preemption. Currently acquired permits continue.
        New ``acquire()`` calls block until active count drops below
        the new max. ``_current_value`` may go negative during drain.

    Invariant: at any moment, the number of "active" (acquired but not
    yet released) permits equals ``max_value - _current_value``. When
    ``_current_value`` is negative, more permits are outstanding than
    the current max allows -- they drain naturally as tasks
    ``release()``.
    """

    def __init__(self, value: int) -> None:
        """Create a semaphore with *value* initial permits.

        Args:
            value: Initial max permits. Must be >= 1.

        Raises:
            ValueError: If value < 1.
        """
        if value < 1:
            raise ValueError(f"value must be >= 1, got {value}")
        self._max_value: int = value
        self._current_value: int = value
        self._waiters: deque[asyncio.Future[None]] = deque()

    @property
    def max_value(self) -> int:
        """Current max permits (may differ from initial after resize)."""
        return self._max_value

    @property
    def active(self) -> int:
        """Number of currently acquired (outstanding) permits.

        Can exceed ``max_value`` temporarily after a resize-down.
        """
        return self._max_value - self._current_value

    async def acquire(self) -> None:
        """Acquire one permit. Blocks if no permits available.

        Raises:
            asyncio.CancelledError: If the waiting coroutine is
                cancelled while blocked.
        """
        # Fast path: permit available and no one queued ahead of us.
        if self._current_value > 0 and not self._waiters:
            self._current_value -= 1
            return

        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        try:
            await fut
        except asyncio.CancelledError:
            # PERMIT-LEAK FIX (V5.1):
            # Three states are possible at this point:
            #   (a) fut not done: nobody granted us a permit yet. Just
            #       remove from the queue.
            #   (b) fut done with result: release()/resize() handed us a
            #       permit (direct hand-off — no _current_value increment
            #       happened). We are about to NOT enter the critical
            #       section, so we MUST give the permit back. Call
            #       release() to wake the next waiter (or restore the
            #       permit to the pool if no waiters remain).
            #   (c) fut already cancelled before we entered the await:
            #       same shape as (a) — `fut in self._waiters` is True
            #       and `self._waiters.remove(fut)` covers it. No permit
            #       was granted, so nothing to give back.
            if fut in self._waiters:
                # Case (a) or (c): pre-grant cancellation. No permit
                # was handed off, so nothing to release.
                self._waiters.remove(fut)
            elif fut.done() and not fut.cancelled() and fut.exception() is None:
                # Case (b): post-grant cancellation. release() handed us
                # a permit via set_result(None) but we're not going to
                # use it. Hand it back so the next waiter (or the pool)
                # gets it.
                self.release()
            raise
        # Granted via direct hand-off from release()/resize().
        # _current_value was NOT decremented (the permit transferred in
        # flight from the previous holder), so we are already accounted
        # for as "active".

    def release(self) -> None:
        """Return a permit. Wakes one blocked waiter if any.

        Direct hand-off semantics: when waiters are queued, the permit
        transfers from the releaser to the next non-cancelled waiter
        without round-tripping through ``_current_value``. This avoids
        a window where two concurrent callers could observe
        ``_current_value > 0`` between waiter-pop and decrement.

        After a resize-down, when no waiters remain, released permits
        that would push ``_current_value`` above ``max_value`` are
        absorbed (clamped). This is correct: the permit belonged to
        the old, larger max.
        """
        # Try to hand the permit directly to the next non-cancelled
        # waiter. The skip-loop drains cancelled waiters whose futures
        # are already done() — protecting against the release/cancel
        # race where a waiter is cancelled mid-iteration.
        while self._waiters:
            # pyre-ignore[1001]: Future is granted via set_result(), not awaited.
            waiter = self._waiters.popleft()
            if not waiter.done():
                # Permit transfers atomically: stays "in flight" with
                # the new owner. Do NOT touch _current_value.
                waiter.set_result(None)
                return
        # No live waiters; restore one permit to the pool (clamped to
        # the current max so resize-down clamps don't drift over).
        self._current_value = min(self._current_value + 1, self._max_value)

    def resize(self, new_max: int) -> None:
        """Change the maximum number of permits.

        Args:
            new_max: New maximum. Must be >= 1.

        When increasing (``new_max > old_max``):
            Additional permits become immediately available. Blocked
            waiters are woken (via direct hand-off) to fill the new
            capacity. Any leftover permits go to the pool, clamped to
            the new max.

        When decreasing (``new_max < old_max``):
            No preemption -- currently active tasks continue.
            ``_current_value`` is reduced by the delta, which may make
            it negative. Future ``acquire()`` calls block until enough
            releases bring ``_current_value`` back above 0.

        Raises:
            ValueError: If new_max < 1.
        """
        if new_max < 1:
            raise ValueError(f"new_max must be >= 1, got {new_max}")
        delta = new_max - self._max_value
        self._max_value = new_max
        if delta > 0:
            # Increase: hand `delta` permits directly to waiters first.
            # Skip cancelled waiters (their futures are already done()).
            granted = 0
            while self._waiters and granted < delta:
                # pyre-ignore[1001]: Future is granted via set_result(), not awaited.
                waiter = self._waiters.popleft()
                if not waiter.done():
                    # Direct hand-off: permit transfers in flight; do
                    # NOT touch _current_value here.
                    waiter.set_result(None)
                    granted += 1
            # Any permits not handed off go to the pool, clamped to max.
            leftover = delta - granted
            if leftover > 0:
                self._current_value = min(
                    self._current_value + leftover,
                    self._max_value,
                )
        elif delta < 0:
            # Decrease: subtract from available pool (may go negative).
            # delta is already negative.
            self._current_value += delta
