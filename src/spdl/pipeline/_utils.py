# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import logging
import sys
import traceback
from asyncio import Task
from collections.abc import Coroutine, Generator
from typing import Any, TypeVar

__all__ = [
    "create_task",
]

_LG: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


# Note:
# This function is intentionally made in a way it cannot be directly attached to
# task callback. Instead it should be wrapped in lambda as follow, even though
# there are some opposition on assigning lambda:
#
# `task.add_done_callback(lambda t: _log_exception(t, stacklevel=2))`
#
# This way, the filename and line number of the log points to the location where
# task was created.
# Otherwise the log will point to the location somewhere deep in `asyncio` module
# which is not very helpful.
def _log_exception(task: Task, stacklevel: int, ignore_cancelled: bool) -> None:
    try:
        task.result()
    except asyncio.exceptions.CancelledError:
        if not ignore_cancelled:
            _LG.warning(
                "Task [%s] was cancelled.", task.get_name(), stacklevel=stacklevel
            )
    except (TimeoutError, asyncio.exceptions.TimeoutError):
        # Timeout does not contain any message
        _LG.error("Task [%s] timeout.", task.get_name(), stacklevel=stacklevel)
    except Exception as err:
        _, _, exc_tb = sys.exc_info()
        f = traceback.extract_tb(exc_tb, limit=-1)[-1]

        _LG.error(
            "Task [%s] failed: %s %s (%s:%d:%s)",
            task.get_name(),
            type(err).__name__,
            err,
            f.filename,
            f.lineno,
            f.name,
            stacklevel=stacklevel,
        )


def create_task(
    coro: Coroutine[Any, Any, T] | Generator[Any, None, T],  # pyre-ignore: [2]
    name: str | None = None,
    ignore_cancelled: bool = True,
) -> Task[T]:
    """Wrapper around :py:func:`asyncio.create_task`. Add logging callback."""
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(
        lambda t: _log_exception(t, stacklevel=3, ignore_cancelled=ignore_cancelled)
    )
    return task
