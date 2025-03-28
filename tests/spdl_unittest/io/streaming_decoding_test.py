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

from ..fixture import get_sample


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
def test_demuxer_has_audio(cmd, expected):
    """has_audio returns true for audio stream"""
    sample = get_sample(cmd)

    with spdl.io.Demuxer(sample.path) as demuxer:
        assert demuxer.has_audio() == expected


def test_demuxer_accept_numpy_array():
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


def test_demuxer_accept_torch_tensor():
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


def test_streaming_video_demuxing_smoke_test():
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


def test_streaming_video_demuxing_parity():
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


def test_demuxer_get_codec():
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -f lavfi -i sine=sample_rate=48000 -af pan='stereo| c0=FR | c1=FR' -frames:v 30 sample.mp4"

    sample = get_sample(cmd)
    demuxer = spdl.io.Demuxer(sample.path)

    audio_codec = demuxer.audio_codec
    video_codec = demuxer.video_codec

    assert audio_codec.name == "aac"
    assert audio_codec.sample_rate == 48000
    assert audio_codec.num_channels == 2
    assert video_codec.name == "h264"
    assert video_codec.width == 320
    assert video_codec.height == 240


def test_decoder_simple():
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 30 sample.mp4"

    sample = get_sample(cmd)

    buffer = spdl.io.load_video(sample.path)
    ref = spdl.io.to_numpy(buffer)

    demuxer = spdl.io.Demuxer(sample.path)
    decoder = spdl.io.Decoder(demuxer.video_codec)

    buffers = []
    for i, packets in enumerate(demuxer.streaming_demux_video(10)):
        frames = decoder.decode(packets)
        print(f"{i}: {frames=}", flush=True)
        buffer = spdl.io.convert_frames(frames)
        buffers.append(spdl.io.to_numpy(buffer))

    frames = decoder.flush()
    print(f"^: {frames=}", flush=True)
    buffer = spdl.io.convert_frames(frames)
    buffers.append(spdl.io.to_numpy(buffer))
    hyp = np.concatenate(buffers)

    np.testing.assert_array_equal(hyp, ref)
