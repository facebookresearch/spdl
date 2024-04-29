import asyncio

import pytest
import spdl.io

import torch


if not spdl.utils.is_nvcodec_available():
    pytest.skip("SPDL is not compiled with NVCODEC support", allow_module_level=True)


CMDS = {
    "video": "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 1000 sample.mp4",
    "image": "ffmpeg -hide_banner -y -f lavfi -i color=0x000000,format=gray -frames:v 1 sample.jpg",
}

CUDA_DEFAULT = 0


@pytest.mark.parametrize("media_type", ["video", "image"])
def test_clone_frames(media_type, get_sample):
    """Cloning frames allows to decode twice"""
    cmd = CMDS[media_type]
    sample = get_sample(cmd)

    async def _test(src):
        frames1 = await spdl.io.async_decode_packets_nvdec(
            await spdl.io.async_demux_media(media_type, src),
            cuda_device_index=CUDA_DEFAULT,
        )
        frames2 = frames1.clone()

        array1 = spdl.io.to_torch(frames1)
        array2 = spdl.io.to_torch(frames2)

        assert torch.all(array1 == array2)

    asyncio.run(_test(sample.path))


@pytest.mark.parametrize("media_type", ["video", "image"])
def test_clone_frames_multi(media_type, get_sample):
    """Can clone multiple times"""
    cmd = CMDS[media_type]
    sample = get_sample(cmd)

    async def _test(src, N=100):
        frames = await spdl.io.async_decode_packets_nvdec(
            await spdl.io.async_demux_media(media_type, src),
            cuda_device_index=CUDA_DEFAULT,
        )
        clones = [frames.clone() for _ in range(N)]

        array = spdl.io.to_torch(frames)
        arrays = [spdl.io.to_torch(c) for c in clones]

        for i in range(N):
            assert torch.all(array == arrays[i])

    asyncio.run(_test(sample.path))
