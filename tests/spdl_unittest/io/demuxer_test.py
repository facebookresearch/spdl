# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import numpy as np
import pytest
import spdl.io
import torch


@pytest.mark.parametrize(
    "cmd,expected",
    [
        (
            "ffmpeg -hide_banner -y -f lavfi -i 'sine=duration=3' -c:a pcm_s16le sample.wav",
            True,
        ),
        (
            "ffmpeg -hide_banner -y -f lavfi -i testsrc -f lavfi -i sine -frames:v 10 sample.mp4",
            True,
        ),
        ("ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 10 sample.mp4", False),
        ("ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg", False),
    ],
)
def test_demuxer_has_audio(get_sample, cmd, expected):
    """has_audio returns true for audio stream"""
    sample = get_sample(cmd)

    with spdl.io.Demuxer(sample.path) as demuxer:
        assert demuxer.has_audio() == expected


def test_demuxer_accept_numpy_array(get_sample):
    """Can instantiate Demuxer with numpy array as source without copying data."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -f lavfi -i sine -frames:v 10 sample.mp4"
    sample = get_sample(cmd)

    with open(sample.path, "rb") as f:
        data = f.read()

    src = np.frombuffer(data, dtype=np.uint8)

    assert np.any(src)
    with spdl.io.Demuxer(src, _zero_clear=True) as demuxer:
        demuxer.demux_video()
    assert not np.any(src)


def test_demuxer_accept_torch_tensor(get_sample):
    """Can instantiate Demuxer with torch tensor as source without copying data."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -f lavfi -i sine -frames:v 10 sample.mp4"
    sample = get_sample(cmd)

    with open(sample.path, "rb") as f:
        data = f.read()

    src = torch.frombuffer(data, dtype=torch.uint8)

    assert torch.any(src)
    with spdl.io.Demuxer(src, _zero_clear=True) as demuxer:
        demuxer.demux_video()
    assert not torch.any(src)


def test_streaming_video_demuxing_smoke_test(get_sample):
    """`streaming_demux_video` can decode packets in streaming fashion."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -f lavfi -i sine -frames:v 10 sample.mp4"
    sample = get_sample(cmd)

    demuxer = spdl.io.Demuxer(sample.path)
    num_packets = 0
    for packets in demuxer.streaming_demux_video(5):
        num_packets += len(packets)

    demuxer = spdl.io.Demuxer(sample.path)
    packets = demuxer.demux_video()

    assert num_packets == len(packets)


def test_streaming_video_demuxing_parity(get_sample):
    """`streaming_demux_video` can decode packets in streaming fashion."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -f lavfi -i sine -frames:v 30 sample.mp4"
    sample = get_sample(cmd)

    def _decode_packets(packets):
        frames = spdl.io.decode_packets(packets)
        buffer = spdl.io.convert_frames(frames)
        return spdl.io.to_numpy(buffer)

    demuxer = spdl.io.Demuxer(sample.path)
    ite = iter(demuxer.streaming_demux_video(30))
    pkts = next(ite)
    hyp = _decode_packets(pkts)

    pkts = spdl.io.demux_video(sample.path)
    ref = _decode_packets(pkts)

    assert np.array_equal(hyp, ref)


def test_demuxer_get_codec(get_sample):
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -f lavfi -i sine -frames:v 30 sample.mp4"

    sample = get_sample(cmd)
    demuxer = spdl.io.Demuxer(sample.path)

    assert demuxer.audio_codec.name == "aac"
    assert demuxer.video_codec.name == "h264"
