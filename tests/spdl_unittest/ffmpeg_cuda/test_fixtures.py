import asyncio

import pytest

import spdl.io


@pytest.fixture
def decode_video_h264_cuvid(get_sample):
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 1000 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)

    def decode_func(cuda_device_index):
        async def _test(cuda_device_index):
            packets = await spdl.io.async_demux("video", sample.path)
            frames = await spdl.io.async_decode_packets(
                packets, cuda_device_index=cuda_device_index, decoder="h264_cuvid"
            )
            buffer = await spdl.io.async_convert_frames(frames)
            return buffer

        return asyncio.run(_test(cuda_device_index))

    return decode_func
