import asyncio

import numpy as np

import pytest
import spdl.io

CMDS = {
    "audio": "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=48000:duration=3' -c:a pcm_s16le sample.wav",
    "video": "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 25 sample.mp4",
    "image": "ffmpeg -hide_banner -y -f lavfi -i color=0x000000,format=gray -frames:v 1 sample.png",
}


async def _load_from_packets(packets):
    frames = await spdl.io.async_decode_packets(packets)
    buffer = await spdl.io.async_convert_frames(frames)
    return spdl.io.to_numpy(buffer)


@pytest.mark.parametrize("media_type", ["audio", "video", "image"])
def test_clone_packets(media_type, get_sample):
    """Cloning packets allows to decode twice"""
    cmd = CMDS[media_type]
    sample = get_sample(cmd)

    async def _test(src):
        packets1 = await spdl.io.async_demux_media(media_type, src)
        packets2 = packets1.clone()

        array1 = await _load_from_packets(packets1)
        array2 = await _load_from_packets(packets2)

        assert np.all(array1 == array2)

    asyncio.run(_test(sample.path))


@pytest.mark.parametrize("media_type", ["audio", "video", "image"])
def test_clone_invalid_packets(media_type, get_sample):
    """Attempt to clone already released packet raises RuntimeError instead of segfault"""
    cmd = CMDS[media_type]
    sample = get_sample(cmd)

    async def _test(src):
        packets = await spdl.io.async_demux_media(media_type, src)
        _ = await spdl.io.async_decode_packets(packets)
        with pytest.raises(RuntimeError):
            packets.clone()

    asyncio.run(_test(sample.path))


@pytest.mark.parametrize("media_type", ["audio", "video", "image"])
def test_clone_packets_multi(media_type, get_sample):
    """Can clone multiple times"""
    cmd = CMDS[media_type]
    sample = get_sample(cmd)

    async def _test(src, N=100):
        packets = await spdl.io.async_demux_media(media_type, src)
        clones = [packets.clone() for _ in range(N)]

        array = await _load_from_packets(packets)
        arrays = [await _load_from_packets(c) for c in clones]

        for i in range(N):
            assert np.all(array == arrays[i])

    asyncio.run(_test(sample.path))
