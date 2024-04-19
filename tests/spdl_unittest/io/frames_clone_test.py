import asyncio

import numpy as np

import pytest
import spdl.io

CMDS = {
    "audio": "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=48000:duration=3' -c:a pcm_s16le sample.wav",
    "video": "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4",
    "image": "ffmpeg -hide_banner -y -f lavfi -i color=0x000000,format=gray -frames:v 1 sample.png",
}


async def _load_from_frames(frames):
    buffer = await spdl.io.async_convert_frames(frames)
    return spdl.io.to_numpy(buffer)


@pytest.mark.parametrize("media_type", ["audio", "video", "image"])
def test_clone_frames(media_type, get_sample):
    """Cloning frames allows to decode twice"""
    cmd = CMDS[media_type]
    sample = get_sample(cmd)

    async def _test(src):
        frames1 = await spdl.io.async_decode_packets(
            await spdl.io.async_demux_media(media_type, src)
        )
        frames2 = frames1.clone()

        array1 = await _load_from_frames(frames1)
        array2 = await _load_from_frames(frames2)

        assert np.all(array1 == array2)

    asyncio.run(_test(sample.path))


@pytest.mark.parametrize("media_type", ["audio", "video", "image"])
def test_clone_invalid_frames(media_type, get_sample):
    """Attempt to clone already released frames raises RuntimeError instead of segfault"""
    cmd = CMDS[media_type]
    sample = get_sample(cmd)

    async def _test(src):
        frames = await spdl.io.async_decode_packets(
            await spdl.io.async_demux_media(media_type, src)
        )
        _ = await spdl.io.async_convert_frames(frames)
        with pytest.raises(RuntimeError):
            frames.clone()

    asyncio.run(_test(sample.path))


@pytest.mark.parametrize("media_type", ["audio", "video", "image"])
def test_clone_frames_multi(media_type, get_sample):
    """Can clone multiple times"""
    cmd = CMDS[media_type]
    sample = get_sample(cmd)

    async def _test(src, N=100):
        frames = await spdl.io.async_decode_packets(
            await spdl.io.async_demux_media(media_type, src)
        )
        clones = [frames.clone() for _ in range(N)]

        array = await _load_from_frames(frames)
        arrays = [await _load_from_frames(c) for c in clones]

        for i in range(N):
            assert np.all(array == arrays[i])

    asyncio.run(_test(sample.path))
