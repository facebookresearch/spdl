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
        for packets in spdl.io.streaming_demux_video(src, timestamps):
            frames = spdl.io.decode_packets(packets)
            buffer = spdl.io.convert_frames(frames)
    
    async def decode(packets):
        frames = await spdl.io.async_decode_packets(packets)
        buffer = await spdl.io.async_convert_frames(frames)
        return buffer

    with spdl.utils.trace_event("async_streaming"):
        tasks = []
        async for packets in spdl.io.async_streaming_demux_video(src, timestamps):
            tasks.append(asyncio.create_task(decode(packets)))

        await asyncio.wait(tasks)


with spdl.utils.tracing("trace_streaming.pftrace"):
    asyncio.run(test())
