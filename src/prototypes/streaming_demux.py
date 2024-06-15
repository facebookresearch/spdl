# pyre-ignore-all-errors

import asyncio

import spdl.io
import spdl.utils


src = "/home/moto/sample.mp4"
timestamps = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
]


async def test():
    with spdl.utils.trace_event("sync_streaming"):
        for _ in spdl.io.streaming_load_video(src, timestamps):
            pass

    with spdl.utils.trace_event("async_streaming"):
        async for _ in spdl.io.async_streaming_load_video(src, timestamps):
            pass


with spdl.utils.tracing("trace_streaming.pftrace"):
    asyncio.run(test())
