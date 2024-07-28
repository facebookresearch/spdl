# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import io

import numpy as np
import pytest

import spdl.io


CMDS = {
    "audio": "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=48000:duration=3' -c:a pcm_s16le sample.wav",
    "video": "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4",
    "image": "ffmpeg -hide_banner -y -f lavfi -i color=0x000000,format=gray -frames:v 1 sample.png",
}


def _is_all_zero(arr):
    return all(int(v) == 0 for v in arr)


@pytest.mark.parametrize("media_type", ["audio", "video", "image"])
def test_demux_bytes_without_copy(media_type, get_sample):
    """Data passed as bytes must be passed without copy."""
    cmd = CMDS[media_type]
    sample = get_sample(cmd)

    demux_func = {
        "audio": spdl.io.async_demux_audio,
        "video": spdl.io.async_demux_video,
        "image": spdl.io.async_demux_image,
    }[media_type]

    async def _test(src):
        assert not _is_all_zero(src)
        _ = await demux_func(src, _zero_clear=True)
        assert _is_all_zero(src)

    with open(sample.path, "rb") as f:
        data = f.read()

    asyncio.run(_test(data))


async def _decode(media_type, src):
    demux_func = {
        "audio": spdl.io.async_demux_audio,
        "video": spdl.io.async_demux_video,
        "image": spdl.io.async_demux_image,
    }[media_type]

    packets = await demux_func(src)
    frames = await spdl.io.async_decode_packets(packets)
    buffer = await spdl.io.async_convert_frames(frames)
    return spdl.io.to_numpy(buffer)


def test_async_decode_audio_bytes(get_sample):
    """audio can be decoded from bytes."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=16000:duration=3' -c:a pcm_s16le sample.wav"
    sample = get_sample(cmd)

    async def _test(path):
        ref = await _decode("audio", path)
        with open(path, "rb") as f:
            hyp = await _decode("audio", f.read())

        assert hyp.shape == (48000, 1)
        assert np.all(ref == hyp)

    asyncio.run(_test(sample.path))


def test_async_decode_video_bytes(get_sample):
    """video can be decoded from bytes."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)

    async def _test(path):
        ref = await _decode("video", path)
        with open(path, "rb") as f:
            hyp = await _decode("video", f.read())

        assert hyp.shape == (1000, 240, 320, 3)
        assert np.all(ref == hyp)

    asyncio.run(_test(sample.path))


def test_demux_image_bytes(get_sample):
    """Image (gray) can be decoded from bytes."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i color=0x000000,format=gray -frames:v 1 sample.png"
    sample = get_sample(cmd, width=320, height=240)

    async def _test(path):
        ref = await _decode("image", path)
        with open(sample.path, "rb") as f:
            hyp = await _decode("image", f.read())

        assert hyp.shape == (240, 320, 3)
        assert np.all(ref == hyp)

    asyncio.run(_test(sample.path))
