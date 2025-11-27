# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "_periodic_dispatch",
    "_StatsCounter",
    "_StageCompleted",
    "_time_str",
]

import asyncio
import time
from asyncio import Task
from collections.abc import Callable, Coroutine, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from spdl.pipeline._common._misc import create_task

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


def is_eof(item: Any) -> bool:
    """Test whether the input item is EOF sentinel value."""
    return bool(item is _EOF)


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


class _StatsCounter:
    def __init__(self) -> None:
        self._n: int = 0
        self._t: float = 0.0

    @property
    def num_items(self) -> int:
        return self._n

    @property
    def ave_time(self) -> float:
        return self._t

    def update(self, t: float, n: int = 1) -> None:
        if n > 0:
            self._n += n
            self._t += (t - self._t) * n / self._n

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


class _StageCompleted(Exception):
    """Notify the pipeline execution system this stage is completed."""

    pass
