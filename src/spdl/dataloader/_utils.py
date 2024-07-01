# pyre-unsafe

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

__all__ = []

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
        _LG.error(
            "Task [%s] failed: %s %s",
            task.get_name(),
            type(err).__name__,
            err,
            stacklevel=stacklevel,
        )
