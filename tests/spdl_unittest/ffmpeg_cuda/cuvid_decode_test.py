import asyncio

import spdl.io

DEFAULT_CUDA = 7


def test_h264_cuvid(get_sample):
    """H264 video can be decoded with h264_cuvid and properly converted to CUDA array."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 1000 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)

    async def _test():
        packets = await spdl.io.async_demux_media("video", sample.path)
        frames = await spdl.io.async_decode_packets(
            packets, cuda_device_index=DEFAULT_CUDA, decoder="h264_cuvid"
        )
        buffer = await spdl.io.async_convert_frames(frames)
        array = spdl.io.to_numba(buffer)
        assert array.shape == (1000, 1, 360, 320)
        assert array.dtype == "uint8"

    asyncio.run(_test())
