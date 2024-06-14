# pyre-unsafe

import asyncio
from concurrent.futures import ThreadPoolExecutor

__all__ = []


def _get_loop(num_workers: int | None) -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    loop.set_default_executor(
        ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="SPDL_BackgroundGenerator",
        )
    )
    return loop
