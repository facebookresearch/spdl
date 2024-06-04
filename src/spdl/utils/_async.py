import asyncio
import functools
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

__all__ = [
    "run_async",
]


def run_async(
    func: Callable[..., T], *args, executor=None, **kwargs
) -> asyncio.Future[T]:
    """Run the given function in the thread pool executor of the current event loop."""
    loop = asyncio.get_running_loop()
    _func = functools.partial(func, *args, **kwargs)
    return loop.run_in_executor(executor, _func)  # pyre-ignore: [6]
