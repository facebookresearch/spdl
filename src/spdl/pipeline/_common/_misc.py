# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Module to stash things common to all the modules internal to SPDL"""

import asyncio
import logging
import os
import sys
import traceback
from asyncio import Task
from collections import defaultdict
from collections.abc import Coroutine, Generator
from typing import Any, TypeVar

__all__ = [
    "create_task",
    "_get_env_bool",
]

_LG: logging.Logger = logging.getLogger(__name__)


T = TypeVar("T")
U = TypeVar("U")


##############################################################################
# Helper function for parsing environment variable
##############################################################################
def _get_env_bool(name: str, default: bool = False) -> bool:
    if name not in os.environ:
        return default

    val = os.environ.get(name, "0")
    trues = ["1", "true", "TRUE", "on", "ON", "yes", "YES"]
    falses = ["0", "false", "FALSE", "off", "OFF", "no", "NO"]
    if val in trues:
        return True
    if val not in falses:
        _LG.warning(
            f"Unexpected environment variable value `{name}={val}`. "
            f"Expected one of {trues + falses}",
            stacklevel=2,
        )
    return False


##############################################################################
# Wrapper for asyncio's create_task.
##############################################################################

# Dictionary to track exception counts by file and line number
_exception_counts: dict[tuple[str, int], int] = defaultdict(int)


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
def _log_exception(
    task: Task,
    stacklevel: int,
    log_cancelled: bool,
    suppress_repeated_logs: bool,
    suppression_threshold: int,
    suppression_warning_interval: int,
    compact: bool,
) -> None:
    try:
        task.result()
    except asyncio.exceptions.CancelledError:
        if log_cancelled:
            _LG.warning(
                "Task [%s] was cancelled.", task.get_name(), stacklevel=stacklevel
            )
    except (TimeoutError, asyncio.exceptions.TimeoutError):
        # Timeout does not contain any message
        _LG.error("Task [%s] timeout.", task.get_name(), stacklevel=stacklevel)
    except Exception as err:
        _, _, exc_tb = sys.exc_info()
        f = traceback.extract_tb(exc_tb, limit=-1)[-1]

        if suppress_repeated_logs and f.filename is not None and f.lineno is not None:
            exception_key = (f.filename, f.lineno)
            _exception_counts[exception_key] += 1
            count = _exception_counts[exception_key]
            if count == suppression_threshold:
                _LG.warning(
                    "Errors are repeated at %s:%d:%s, holding on logging.",
                    f.filename,
                    f.lineno,
                    f.name,
                )
                return
            elif count > suppression_threshold:
                if count % suppression_warning_interval == 0:
                    _LG.warning(
                        "%d errors were logged at %s:%d:%s.",
                        count,
                        f.filename,
                        f.lineno,
                        f.name,
                    )
                    return
                else:
                    return

        if compact:
            _LG.error(
                "Task [%s]: %s: %s (%s:%d:%s)",
                task.get_name(),
                type(err).__name__,
                err,
                f.filename,
                f.lineno,
                f.name,
                stacklevel=stacklevel,
            )
        else:
            _LG.exception("Task [%s] failed.", task.get_name(), stacklevel=stacklevel)


def create_task(
    coro: Coroutine[Any, Any, T] | Generator[Any, None, T],
    name: str | None = None,
    log_cancelled: bool = False,
    suppress_repeated_logs: bool = False,
    suppression_threshold: int = 5,
    suppression_warning_interval: int = 100,
) -> Task[T]:
    """Wrap :py:func:`asyncio.create_task` and add logging callback.

    Args:
        coro: The coroutine or generator to be executed as a task.
        name: Optional name for the task. If ``None``, asyncio will generate a default name.
        log_cancelled: If ``True``, log a warning when the task is cancelled.
            Defaults to ``False``.
        suppress_repeated_logs: If ``True``, suppress repeated exception logs from the same
            location after reaching the suppression threshold.
            Defaults to ``False``.
        suppression_threshold: Number of exceptions from the same location before suppression
            kicks in.
            Defaults to ``5``.
        suppression_warning_interval: Interval at which to log suppression warnings after
            threshold is reached.
            Defaults to ``100``.

        Returns:
            Task[T]: The created asyncio Task with exception logging callback attached.
    """
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(
        lambda t: _log_exception(
            t,
            stacklevel=3,
            log_cancelled=log_cancelled,
            suppress_repeated_logs=suppress_repeated_logs,
            suppression_threshold=suppression_threshold,
            suppression_warning_interval=suppression_warning_interval,
            compact=False,
        )
    )
    return task
