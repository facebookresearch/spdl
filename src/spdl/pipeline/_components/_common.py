# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "_drive_to_completion",
    "_EOF",
    "_EPOCH_END",
    "_periodic_dispatch",
    "_P2Percentile",
    "_ShieldedHook",
    "_SKIP",
    "_StatsCounter",
    "_time_str",
    "is_eof",
    "is_epoch_end",
    "StageInfo",
]

import asyncio
import time
from asyncio import Task
from collections.abc import Callable, Coroutine, Iterator
from contextlib import contextmanager
from typing import Any, AsyncContextManager, TypeVar

from spdl.pipeline._common._misc import create_task
from spdl.pipeline._common._types import StageInfo as StageInfo  # noqa: F811

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
_EPOCH_END = _Sentinel("EPOCH_END")  # Indicate the end of one epoch.
_SKIP: None = None


def is_eof(item: Any) -> bool:
    """Test whether the input item is EOF sentinel value."""
    return bool(item is _EOF)


def is_epoch_end(item: Any) -> bool:
    """Test whether the input item is an epoch-end sentinel value.

    Custom aggregators can use this to handle epoch boundaries,
    for example to flush or discard partial batches.
    """
    return bool(item is _EPOCH_END)


def _time_str(val: float) -> str:
    if val < 0.0001:
        val *= 1e6
        unit = "us"
    elif val < 1:
        val *= 1e3
        unit = "ms"
    else:
        unit = "sec"

    return f"{val:6.1f} [{unit:>3s}]"


class _P2Percentile:
    """Streaming percentile estimation using the P-square algorithm.

    Maintains O(1) memory (5 markers) and O(1) per update.
    Reference: Jain & Chlamtac, "The P-Square Algorithm for Dynamic
    Calculation of Percentiles and Histograms without Storing
    Observations", 1985.
    """

    def __init__(self, p: float) -> None:
        self._p: float = p / 100.0
        self._count: int = 0
        self._q: list[float] = [0.0] * 5
        self._n: list[int] = [0] * 5
        self._np: list[float] = [0.0] * 5
        self._dn: list[float] = [0.0] * 5

    @property
    def value(self) -> float:
        if self._count == 0:
            return 0.0
        if self._count < 5:
            sorted_data = sorted(self._q[: self._count])
            idx = min(int(self._count * self._p), self._count - 1)
            return sorted_data[idx]
        return self._q[2]

    def update(self, x: float) -> None:
        self._count += 1

        if self._count <= 5:
            self._q[self._count - 1] = x
            if self._count == 5:
                self._initialize()
            return

        self._process(x)

    def reset(self) -> None:
        self._count = 0
        self._q = [0.0] * 5
        self._n = [0] * 5
        self._np = [0.0] * 5
        self._dn = [0.0] * 5

    def _initialize(self) -> None:
        self._q.sort()
        p = self._p
        self._n = [0, 1, 2, 3, 4]
        self._np = [0.0, 2.0 * p, 4.0 * p, 2.0 + 2.0 * p, 4.0]
        self._dn = [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0]

    def _process(self, x: float) -> None:
        if x < self._q[0]:
            self._q[0] = x
            k = 0
        elif x < self._q[1]:
            k = 0
        elif x < self._q[2]:
            k = 1
        elif x < self._q[3]:
            k = 2
        elif x <= self._q[4]:
            k = 3
        else:
            self._q[4] = x
            k = 3

        for i in range(k + 1, 5):
            self._n[i] += 1

        for i in range(5):
            self._np[i] += self._dn[i]

        for i in range(1, 4):
            d = self._np[i] - self._n[i]
            if (d >= 1 and self._n[i + 1] - self._n[i] > 1) or (
                d <= -1 and self._n[i - 1] - self._n[i] < -1
            ):
                d_sign = 1 if d > 0 else -1
                qi = self._parabolic(i, d_sign)
                if self._q[i - 1] < qi < self._q[i + 1]:
                    self._q[i] = qi
                else:
                    self._q[i] = self._linear(i, d_sign)
                self._n[i] += d_sign

    def _parabolic(self, i: int, d: int) -> float:
        qi = self._q[i]
        ni = self._n[i]
        nim1 = self._n[i - 1]
        nip1 = self._n[i + 1]
        left = (ni - nim1 + d) * (self._q[i + 1] - qi) / (nip1 - ni)
        right = (nip1 - ni - d) * (qi - self._q[i - 1]) / (ni - nim1)
        return qi + d / (nip1 - nim1) * (left + right)

    def _linear(self, i: int, d: int) -> float:
        return self._q[i] + d * (self._q[i + d] - self._q[i]) / (
            self._n[i + d] - self._n[i]
        )


class _StatsCounter:
    def __init__(self) -> None:
        self._n: int = 0
        self._t: float = 0.0
        self._p90 = _P2Percentile(90)
        self._p99 = _P2Percentile(99)
        self._lap_p90 = _P2Percentile(90)
        self._lap_p99 = _P2Percentile(99)

    @property
    def num_items(self) -> int:
        return self._n

    @property
    def ave_time(self) -> float:
        return self._t

    @property
    def p90_time(self) -> float:
        return self._p90.value

    @property
    def p99_time(self) -> float:
        return self._p99.value

    def consume_lap_percentiles(self) -> tuple[float, float]:
        p90 = self._lap_p90.value
        p99 = self._lap_p99.value
        self._lap_p90.reset()
        self._lap_p99.reset()
        return p90, p99

    def update(self, t: float, n: int = 1) -> None:
        if n > 0:
            self._n += n
            self._t += (t - self._t) * n / self._n
            self._p90.update(t)
            self._p99.update(t)
            self._lap_p90.update(t)
            self._lap_p99.update(t)

    @contextmanager
    def count(self) -> Iterator[None]:
        t0 = time.monotonic()
        yield
        elapsed = time.monotonic() - t0
        self.update(elapsed)


async def _periodic_dispatch(
    afun: Callable[[], Coroutine[None, None, None]],
    done: asyncio.Event,
    interval: float,
) -> None:
    assert interval > 0, "[InternalError] `interval` must be greater than 0."
    pending: set[Task] = set()
    target = time.monotonic() + interval
    while True:
        if (dt := target - time.monotonic()) > 0:
            await asyncio.wait([create_task(done.wait())], timeout=dt)

        if done.is_set():
            break

        target = time.monotonic() + interval
        pending.add(create_task(afun()))

        # Assumption interval >> task duration.
        _, pending = await asyncio.wait(pending, return_when="FIRST_COMPLETED")

    if pending:
        await asyncio.wait(pending)


async def _drive_to_completion(
    coro: Coroutine[Any, Any, T],
) -> tuple[T, asyncio.CancelledError | None]:
    """Drive ``coro`` to completion, absorbing repeated cancellations.

    A task can be cancelled more than once: each ``cancel()`` queues another
    ``CancelledError`` for the next suspension point. A single ``asyncio.shield``
    only absorbs one, so we loop until the shielded task actually finishes.

    Returns the coroutine's result together with the first ``CancelledError``
    seen (or ``None``). The caller is responsible for re-raising the
    cancellation once any remaining shielded work is done — it is never
    swallowed here. A non-cancellation exception from ``coro`` propagates
    immediately and takes priority over a pending cancellation.

    The surrounding task may be cancelled repeatedly (e.g. the pipeline
    executor re-cancels orphaned stages on every loop iteration). Each delivery
    interrupts our ``await`` with ``CancelledError`` even after ``coro`` itself
    has already finished. We disambiguate via ``task.cancelled()``: if ``coro``
    was itself cancelled, that is its result and we propagate it; otherwise the
    ``CancelledError`` is a cancellation of *our* wait, so we keep ``coro``'s
    real result (value or non-cancel exception) and report the cancellation for
    the caller to handle.
    """
    task = asyncio.ensure_future(coro)
    cancelled: asyncio.CancelledError | None = None
    while True:
        try:
            return await asyncio.shield(task), cancelled
        except asyncio.CancelledError as e:
            cancelled = cancelled or e
            if task.done():
                if task.cancelled():
                    raise
                # ``coro`` finished; ``task.result()`` returns its value or
                # re-raises its (non-cancel) exception.
                return task.result(), cancelled


class _ShieldedHook:
    """Wrap a stage-hook context manager so ``__aenter__``/``__aexit__`` run to
    completion even when the surrounding task is cancelled.

    The cancellation is re-raised once the shielded enter/exit finishes, so the
    stage still shuts down — it is only deferred, not dropped. This protects
    finalization that must not be lost (e.g. flushing the final task stats in
    ``TaskStatsHook.stage_hook`` or the queue stats in ``StatsQueue.stage_hook``).

    .. caution::

       This makes the hook's enter/exit uninterruptible. Only wrap finalization
       that is trusted to terminate; a ``stage_hook`` that blocks after ``yield``
       would make the whole stage impossible to cancel.
    """

    def __init__(self, cm: AsyncContextManager[Any]) -> None:
        self._cm = cm

    async def __aenter__(self) -> Any:
        res, cancelled = await _drive_to_completion(self._cm.__aenter__())
        if cancelled is None:
            return res
        # Entered successfully but we were cancelled; release before re-raising
        # so a half-initialized hook is not leaked. The exit is shielded too.
        await _drive_to_completion(self._cm.__aexit__(None, None, None))
        raise cancelled

    async def __aexit__(self, *exc_info: Any) -> Any:
        result, cancelled = await _drive_to_completion(self._cm.__aexit__(*exc_info))
        # Surface a deferred cancellation only when there is no original
        # exception to preserve, or the hook suppressed it (truthy ``result``).
        # When an exception is already propagating into this exit (e.g. a stage
        # failure) and the hook did not suppress it, that exception must win —
        # a cancellation that arrived during finalization must not mask it.
        # ``result`` is the hook's exception-suppression signal, which we also
        # return to honor the context-manager contract.
        if cancelled is not None and (exc_info[0] is None or result):
            raise cancelled
        return result
