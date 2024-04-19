import asyncio

import pytest
import spdl.io

import torch

CMDS = {
    "video": "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 1000 sample.mp4",
    "image": "ffmpeg -hide_banner -y -f lavfi -i color=0x000000,format=gray -frames:v 1 sample.jpg",
}

CUDA_DEFAULT = 0


async def _load_from_frames(frames):
    buffer = await spdl.io.async_convert_frames(frames)
    return spdl.io.to_torch(buffer)


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

        array1 = await _load_from_frames(frames1)
        array2 = await _load_from_frames(frames2)

        assert torch.all(array1 == array2)

    asyncio.run(_test(sample.path))


@pytest.mark.parametrize("media_type", ["video", "image"])
def test_clone_frames_after_conversion(media_type, get_sample):
    """NvDec frames can be cloned even after conversion

    Unlike FFmpegFrames, NvDecFrames share the buffer between Buffer class
    and Frames class, because NvDecFrames class allocates CUDA memory directly
    during the decoding.
    For this reason, `convert_frames` function does not disconnect the lifetime
    of the buffer and frame Python variable.

    This is just an implementation detail and not a spec, but testing it here
    as a reminder just in case the behavior is changed.
    """
    cmd = CMDS[media_type]
    sample = get_sample(cmd)

    async def _test(src):
        frames = await spdl.io.async_decode_packets_nvdec(
            await spdl.io.async_demux_media(media_type, src),
            cuda_device_index=CUDA_DEFAULT,
        )
        _ = await spdl.io.async_convert_frames(frames)
        frames.clone()  # does not raise

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

        array = await _load_from_frames(frames)
        arrays = [await _load_from_frames(c) for c in clones]

        for i in range(N):
            assert torch.all(array == arrays[i])

    asyncio.run(_test(sample.path))
