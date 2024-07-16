# pyre-unsafe

import asyncio
import concurrent.futures
import logging
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

from threading import Thread

__all__ = [
    "create_task",
]

_LG = logging.getLogger(__name__)


def _get_loop(num_workers: int | None) -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    loop.set_default_executor(
        ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="SPDL_BackgroundGenerator",
        )
    )
    return loop


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
def _log_exception(task, stacklevel):
    try:
        task.result()
    except asyncio.exceptions.CancelledError:
        _LG.warning("Task [%s] was cancelled.", task.get_name(), stacklevel=stacklevel)
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


def create_task(coro, name=None) -> asyncio.Task:
    """Wrapper around :py:func:`asyncio.create_task`. Add logging callback."""
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(lambda t: _log_exception(t, stacklevel=3))
    return task


##############################################################################
# The utilities for running loop in a thread and stopping it from another thread
##############################################################################


class _EventLoopThread(Thread):
    def __init__(self, loop):
        super().__init__()
        self.loop = loop

    def run(self):
        self.loop.run_forever()


def _run_coro(loop, coro, timeout=None):
    try:
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout)
    except concurrent.futures.TimeoutError:
        raise TimeoutError("Failed to execute coroutine") from None


# Note:
#
# The following code are based off of
# https://github.com/python/cpython/blob/3.10/Lib/asyncio/runners.py


async def _cancel(tasks):
    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)

    loop = asyncio.get_running_loop()
    for task in tasks:
        if task.cancelled():
            continue

        if (err := task.exception()) is not None:
            loop.call_exception_handler(
                {
                    "message": "unhandled exception during asyncio.run() shutdown",
                    "exception": err,
                    "task": task,
                }
            )


def _cancel_all_tasks(loop):
    # Note: `asyncio.all_tasks` must be called outside of loop.
    # otherwise it includes the currently running coroutine.
    tasks = asyncio.all_tasks(loop)
    if not tasks:
        return

    _LG.debug("Cancelling %d tasks to cancel.", len(tasks))
    _run_coro(loop, _cancel(tasks))


def _stop_loop(loop):
    try:
        _cancel_all_tasks(loop)
        _LG.debug("Shutting down asyncgens")
        _run_coro(loop, loop.shutdown_asyncgens())
        _LG.debug("Shutting executors")
        _run_coro(loop, loop.shutdown_default_executor())
    finally:
        _LG.debug("stopping loop")
        loop.call_soon_threadsafe(loop.stop)


##############################################################################
