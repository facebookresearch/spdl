import asyncio

import spdl

DEFAULT_CUDA = 0


def test_decode_video_nvdec(get_sample):
    """Can decode video with NVDEC"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 1000 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)

    timestamps = [(i, i + 1) for i in range(10)]

    async def _test():
        decode_tasks = []
        async for packets in spdl.async_demux_video(sample.path, timestamps=timestamps):
            print(packets)
            filtered = await spdl.async_apply_bsf(packets)
            print(filtered)
            decode_tasks.append(
                spdl.async_decode_nvdec(filtered, cuda_device_index=DEFAULT_CUDA)
            )
        results = await asyncio.gather(*decode_tasks)
        for frames in results:
            print(frames)

    asyncio.run(_test())


def test_decode_image_nvdec(get_sample):
    """Can decode image with NVDEC"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
    sample = get_sample(cmd, width=320, height=240)

    async def _test():
        packets = await spdl.async_demux_image(sample.path)
        print(packets)
        frames = await spdl.async_decode_nvdec(packets, cuda_device_index=DEFAULT_CUDA)
        print(frames)

    asyncio.run(_test())
