# pyre-unsafe

import asyncio
import concurrent.futures
import logging
import sys
import traceback
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor

from threading import Event as SyncEvent, Thread

__all__ = [
    "create_task",
]

_LG = logging.getLogger(__name__)


def _get_loop(num_workers: int | None) -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    loop.set_default_executor(
        ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="spdl_",
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
def _log_exception(task, stacklevel, ignore_cancelled):
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
    coro, name: str | None = None, ignore_cancelled: bool = True
) -> asyncio.Task:
    """Wrapper around :py:func:`asyncio.create_task`. Add logging callback."""
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(
        lambda t: _log_exception(t, stacklevel=3, ignore_cancelled=ignore_cancelled)
    )
    return task


##############################################################################
# _EventLoop (Thread)
##############################################################################


# Note:
# This class has a bit exessive debug logs, because it is tricky to debug
# it from the outside.
class _EventLoop:
    def __init__(self, coro: Coroutine[None, None, None], num_threads: int):
        self._coro = coro
        self._num_threads = num_threads

        self._loop = None

        self._task_started = SyncEvent()
        self._task_completed = SyncEvent()
        self._stop_requested = SyncEvent()

        self._thread = Thread(target=lambda: asyncio.run(self._execute_task()))

    def __str__(self):
        return str(
            {
                "thread_alive": self._thread.is_alive(),
                "task_started": self._task_started.is_set(),
                "task_completed": self._task_completed.is_set(),
                "stop_requested": self._stop_requested.is_set(),
            }
        )

    async def _execute_task(self) -> None:
        _LG.debug("The event loop thread coroutine is started.")
        _LG.debug("Initializing the thread pool of size=%d.", self._num_threads)
        self._loop = asyncio.get_running_loop()
        self._loop.set_default_executor(
            ThreadPoolExecutor(
                max_workers=self._num_threads,
                thread_name_prefix="spdl_",
            )
        )

        _LG.debug("Starting the task.")

        task = create_task(self._coro, name="Pipeline::main")
        task.add_done_callback(lambda t: self._task_completed.set())

        self._task_started.set()
        while not task.done():
            await asyncio.wait([task], timeout=0.1)

            if not task.done() and self._stop_requested.is_set():
                _LG.debug(
                    "Stop request is received, but the task is not complete. "
                    "Cancelling the task."
                )
                task.cancel()
                await asyncio.wait([task])

        _LG.debug("The task is completed.")

        _LG.debug("%s", self)
        if not self._stop_requested.is_set():
            _LG.debug("Keep the event loop alive until the stop request is made.")
            while not self._stop_requested.is_set():
                await asyncio.sleep(0.1)
        _LG.debug("The background task is completed.")
        _LG.debug("The event loop is now shutdown.")

    def start(self, *, timeout: float | None = None) -> None:
        """Start the thread and block until the loop is initialized."""
        _LG.debug("Starting the event loop thread.")
        self._thread.start()
        _LG.debug("Waiting for the loop to be initialized.")
        self._task_started.wait(timeout=timeout)
        _LG.debug("The event loop thread is initialized.")

    def is_started(self) -> bool:
        """Check if the event loop thread is started."""
        return self._task_started.is_set()

    def is_task_completed(self) -> bool:
        """Check if the task is completed."""
        return self._task_completed.is_set()

    def stop(self):
        """Issue loop stop request and wait for the thread to join."""
        if not self._stop_requested.is_set():
            _LG.debug("Requesting the event loop thread to stop.")
            self._stop_requested.set()

    def join(self, *, timeout: float | None = None) -> None:
        if not self._stop_requested.is_set():
            raise RuntimeError(
                "The event loop thread is not stopped. Call stop() first."
            )

        _LG.debug("Waiting for the event loop thread to join.")
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError(f"Thread did not join after {timeout} seconds.")
        _LG.debug("The event loop thread joined.")

    def run_coroutine_threadsafe(self, coro) -> concurrent.futures.Future:
        """Call coroutine in the loop thread."""
        if not self._task_started.is_set():
            raise RuntimeError("Event loop is not started.")
        if not self._loop.is_running():
            raise RuntimeError("Event loop is not running.")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)
