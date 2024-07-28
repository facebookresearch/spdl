# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

import numpy as np

import pytest
import spdl.io


def test_streaming_decode(get_sample):
    """Streaming decode yields the same frames as batched decode"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 50 sample.mp4"
    sample = get_sample(cmd)

    async def test(src):
        packets = await spdl.io.async_demux_video(src)
        frames = await spdl.io.async_decode_packets(packets.clone())
        buffer = await spdl.io.async_convert_frames(frames)
        ref_array = spdl.io.to_numpy(buffer)

        agen = spdl.io.async_streaming_decode_packets(packets, 1)
        for i in range(50):
            print(i)
            frame = await anext(agen)
            buffer = await spdl.io.async_convert_frames(frame)
            array = spdl.io.to_numpy(buffer)
            print(f"{ref_array.shape=}, {array.shape=}")
            assert np.array_equal(ref_array[i : i + 1], array)

    asyncio.run(test(sample.path))


def test_streaming_decode_indivisible(get_sample):
    """Streaming decode can handle the number of frames that's not divisble to batch size"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 8 sample.mp4"
    sample = get_sample(cmd)

    async def test(src):
        packets = await spdl.io.async_demux_video(src)
        agen = spdl.io.async_streaming_decode_packets(packets, 5)

        frame = await anext(agen)
        buffer = await spdl.io.async_convert_frames(frame)
        array = spdl.io.to_numpy(buffer)
        assert array.shape == (5, 240, 320, 3)

        frame = await anext(agen)
        buffer = await spdl.io.async_convert_frames(frame)
        array = spdl.io.to_numpy(buffer)
        assert array.shape == (3, 240, 320, 3)

    asyncio.run(test(sample.path))


def test_streaming_decode_stop_iteration(get_sample):
    """Calling anext over and over again after the generator exhausted should not fail"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.mp4"
    sample = get_sample(cmd)

    async def test(src):
        packets = await spdl.io.async_demux_video(src)
        agen = spdl.io.async_streaming_decode_packets(packets, 5)

        frame = await anext(agen)
        buffer = await spdl.io.async_convert_frames(frame)
        array = spdl.io.to_numpy(buffer)
        assert array.shape == (1, 240, 320, 3)

        for _ in range(20):
            with pytest.raises(StopAsyncIteration):
                await anext(agen)

    asyncio.run(test(sample.path))


def test_streaming_decode_carryover(get_sample):
    """Streaming decode can handle the carry over from previous decoding"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 30 sample.mp4"
    sample = get_sample(cmd)

    # Increase the frame rate from 30 to 1000 fps.
    # this will internally cause the filtergraph to duplicate the frames.
    filter_desc = "fps=1000"

    async def test(src):
        packets = await spdl.io.async_demux_video(src)
        frames = await spdl.io.async_decode_packets(
            packets.clone(), filter_desc=filter_desc
        )
        buffer = await spdl.io.async_convert_frames(frames)
        ref_array = spdl.io.to_numpy(buffer)
        print(ref_array.shape)

        agen = spdl.io.async_streaming_decode_packets(
            packets, 5, filter_desc=filter_desc
        )

        for i in range(200):
            print(i)
            frame = await anext(agen)
            buffer = await spdl.io.async_convert_frames(frame)
            array = spdl.io.to_numpy(buffer)
            assert array.shape == (5, 3, 240, 320)

    asyncio.run(test(sample.path))
