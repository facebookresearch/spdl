import asyncio
import time

import numpy as np

import pytest
import spdl.io

from spdl.lib import _libspdl

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

    demux_func = {
        "audio": spdl.io.async_demux_audio,
        "video": spdl.io.async_demux_video,
        "image": spdl.io.async_demux_image,
    }[media_type]

    async def _test(src):
        packets1 = await demux_func(src)
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

    demux_func = {
        "audio": spdl.io.async_demux_audio,
        "video": spdl.io.async_demux_video,
        "image": spdl.io.async_demux_image,
    }[media_type]

    async def _test(src):
        packets = await demux_func(src)
        _ = await spdl.io.async_decode_packets(packets)
        with pytest.raises(TypeError):
            packets.clone()

    asyncio.run(_test(sample.path))


@pytest.mark.parametrize("media_type", ["audio", "video", "image"])
def test_clone_packets_multi(media_type, get_sample):
    """Can clone multiple times"""
    cmd = CMDS[media_type]
    sample = get_sample(cmd)

    demux_func = {
        "audio": spdl.io.async_demux_audio,
        "video": spdl.io.async_demux_video,
        "image": spdl.io.async_demux_image,
    }[media_type]

    async def _test(src, N=100):
        packets = await demux_func(src)
        clones = [packets.clone() for _ in range(N)]

        array = await _load_from_packets(packets)
        arrays = [await _load_from_packets(c) for c in clones]

        for i in range(N):
            assert np.all(array == arrays[i])

    asyncio.run(_test(sample.path))


def test_sample_decoding_time(get_sample):
    """Sample decoding works"""
    # https://stackoverflow.com/questions/63725248/how-can-i-set-gop-size-to-be-a-multiple-of-the-input-framerate
    cmd = (
        "ffmpeg -hide_banner -y -f lavfi -i testsrc "
        "-force_key_frames 'expr:eq(mod(n, 25), 0)' "
        "-frames:v 5000 sample.mp4"
    )
    # Note: You can use the following command to check that the generated video has the keyframes
    # at the expected positions:
    # Ref: https://www.reddit.com/r/ffmpeg/comments/k6su5f/how_can_i_get_an_output_of_all_keyframe/
    # Use ffprobe -loglevel error -select_streams v:0 -show_entries packet=pts_time,flags -of csv=print_section=0 sample.mp4 | grep K__
    sample = get_sample(cmd)

    async def _test(path):
        indices = list(range(0, 5000, 100))

        packets = await spdl.io.async_demux_video(path)
        t0 = time.monotonic()
        frames = await spdl.io.async_decode_packets(packets.clone())
        frames = frames[indices]
        elapsed_ref = time.monotonic() - t0
        buffer = await spdl.io.async_convert_frames(frames)
        array_ref = spdl.io.to_numpy(buffer)

        t0 = time.monotonic()
        frames = await spdl.io.async_sample_decode_video(packets, indices)
        elapsed = time.monotonic() - t0
        buffer = await spdl.io.async_convert_frames(frames)
        array = spdl.io.to_numpy(buffer)

        print(f"{elapsed_ref=}, {elapsed=}")
        assert np.all(array == array_ref)

        # should be much faster than 2x
        assert elapsed_ref / 2 > elapsed

    asyncio.run(_test(sample.path))
