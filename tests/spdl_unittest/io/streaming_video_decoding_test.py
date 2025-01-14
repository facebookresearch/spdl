# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import numpy as np
import pytest
import spdl.io


def test_streaming_decode(get_sample):
    """Streaming decode yields the same frames as batched decode"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 50 sample.mp4"
    sample = get_sample(cmd)

    packets = spdl.io.demux_video(sample.path)
    frames = spdl.io.decode_packets(packets.clone())
    buffer = spdl.io.convert_frames(frames)
    ref_array = spdl.io.to_numpy(buffer)

    gen = spdl.io.streaming_decode_packets(packets, 1)
    for i in range(50):
        print(i)
        frame = next(gen)
        buffer = spdl.io.convert_frames(frame)
        array = spdl.io.to_numpy(buffer)
        print(f"{ref_array.shape=}, {array.shape=}")
        assert np.array_equal(ref_array[i : i + 1], array)


def test_streaming_decode_indivisible(get_sample):
    """Streaming decode can handle the number of frames that's not divisble to batch size"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 8 sample.mp4"
    sample = get_sample(cmd)

    packets = spdl.io.demux_video(sample.path)
    gen = spdl.io.streaming_decode_packets(packets, 5)

    frame = next(gen)
    buffer = spdl.io.convert_frames(frame)
    array = spdl.io.to_numpy(buffer)
    assert array.shape == (5, 240, 320, 3)

    frame = next(gen)
    buffer = spdl.io.convert_frames(frame)
    array = spdl.io.to_numpy(buffer)
    assert array.shape == (3, 240, 320, 3)


def test_streaming_decode_stop_iteration(get_sample):
    """Calling anext over and over again after the generator exhausted should not fail"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.mp4"
    sample = get_sample(cmd)

    packets = spdl.io.demux_video(sample.path)
    gen = spdl.io.streaming_decode_packets(packets, 5)

    frame = next(gen)
    buffer = spdl.io.convert_frames(frame)
    array = spdl.io.to_numpy(buffer)
    assert array.shape == (1, 240, 320, 3)

    for _ in range(20):
        with pytest.raises(StopIteration):
            next(gen)


def test_streaming_decode_carryover(get_sample):
    """Streaming decode can handle the carry over from previous decoding"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 30 sample.mp4"
    sample = get_sample(cmd)

    # Increase the frame rate from 30 to 1000 fps.
    # this will internally cause the filtergraph to duplicate the frames.
    filter_desc = "fps=1000"

    packets = spdl.io.demux_video(sample.path)
    frames = spdl.io.decode_packets(packets.clone(), filter_desc=filter_desc)
    buffer = spdl.io.convert_frames(frames)
    ref_array = spdl.io.to_numpy(buffer)
    print(ref_array.shape)

    gen = spdl.io.streaming_decode_packets(packets, 5, filter_desc=filter_desc)

    for i in range(200):
        print(i)
        frame = next(gen)
        buffer = spdl.io.convert_frames(frame)
        array = spdl.io.to_numpy(buffer)
        assert array.shape == (5, 3, 240, 320)
