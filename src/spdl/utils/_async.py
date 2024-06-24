# pyre-unsafe

import asyncio
import functools
import logging
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar


__all__ = ["run_async"]

_LG = logging.getLogger(__name__)

T = TypeVar("T")


async def run_async(
    func: Callable[..., T],
    *args,
    _executor: ThreadPoolExecutor | None = None,
    **kwargs,
) -> Awaitable[T]:
    """Run the given synchronous function asynchronously (in a thread).

    .. note::

       To achieve the true concurrency, the function must be thread-safe and must
       release the GIL.

    Args:
        func: The function to run.
        args: Positional arguments to the ``func``.
        _executor: Custom executor.
            If ``None`` the default executor of the current event loop is used.
        kwargs: Keyword arguments to the ``func``.
    """
    loop = asyncio.get_running_loop()
    _func = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(_executor, _func)  # pyre-ignore: [6]
