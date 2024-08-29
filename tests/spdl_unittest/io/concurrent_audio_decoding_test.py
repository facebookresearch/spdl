# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

import numpy as np
import pytest

import spdl.io
import spdl.utils
from spdl.io import get_audio_filter_desc, get_filter_desc


def _decode_audio(src, sample_fmt=None):
    buffer = asyncio.run(
        spdl.io.async_load_audio(
            src, filter_desc=get_audio_filter_desc(sample_fmt=sample_fmt)
        )
    )
    return spdl.io.to_numpy(buffer)


@pytest.mark.parametrize(
    "sample_fmts",
    [("s16p", "int16"), ("s16", "int16"), ("fltp", "float32"), ("flt", "float32")],
)
def test_audio_buffer_conversion_s16p(sample_fmts, get_sample):
    # fmt: off
    cmd = """
    ffmpeg -hide_banner -y \
    -f lavfi -i 'sine=sample_rate=8000:frequency=305:duration=5' \
    -f lavfi -i 'sine=sample_rate=8000:frequency=300:duration=5' \
    -filter_complex amerge  -c:a pcm_s16le sample.wav
    """
    # fmt: on
    sample = get_sample(cmd)

    sample_fmt, expected = sample_fmts
    array = _decode_audio(src=sample.path, sample_fmt=sample_fmt)

    assert array.ndim == 2
    assert array.dtype == np.dtype(expected)
    shape = (2, 40000) if sample_fmt.endswith("p") else (40000, 2)
    assert array.shape == shape


def test_batch_audio_conversion(get_sample):
    # fmt: off
    cmd = """
    ffmpeg -hide_banner -y \
    -f lavfi -i 'sine=sample_rate=8000:frequency=305:duration=5' \
    -f lavfi -i 'sine=sample_rate=8000:frequency=300:duration=5' \
    -filter_complex amerge  -c:a pcm_s16le sample.wav
    """
    # fmt: on
    sample = get_sample(cmd)

    timestamps = [(0, 1), (1, 1.5), (2, 2.7)]

    async def _test():
        decoding = []

        demuxer = spdl.io.Demuxer(sample.path)
        for ts in timestamps:
            packets = demuxer.demux_audio(ts)
            filter_desc = get_filter_desc(packets, num_frames=8_000)
            coro = spdl.io.async_decode_packets(packets, filter_desc=filter_desc)
            decoding.append(asyncio.create_task(coro))

        frames = await asyncio.gather(*decoding)

        buffer = await spdl.io.async_convert_frames(frames)
        array = spdl.io.to_numpy(buffer)
        return array

    array = asyncio.run(_test())

    assert array.shape == (3, 2, 8000)
