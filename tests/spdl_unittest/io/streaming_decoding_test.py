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

from ..fixture import FFMPEG_CLI, get_sample


@pytest.mark.parametrize(
    "cmd,expected",
    [
        (
            f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i 'sine=duration=3' -c:a pcm_s16le sample.wav",
            True,
        ),
        (
            f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -f lavfi -i sine -frames:v 10 sample.mp4",
            True,
        ),
        (
            f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 10 sample.mp4",
            False,
        ),
        (
            f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg",
            False,
        ),
    ],
)
def test_demuxer_has_audio(cmd, expected):
    """has_audio returns true for audio stream"""
    sample = get_sample(cmd)

    with spdl.io.Demuxer(sample.path) as demuxer:
        assert demuxer.has_audio() == expected


def test_demuxer_accept_numpy_array():
    """Can instantiate Demuxer with numpy array as source without copying data."""
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -f lavfi -i sine -frames:v 10 sample.mp4"
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
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -f lavfi -i sine -frames:v 10 sample.mp4"
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
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -f lavfi -i sine -frames:v 10 sample.mp4"
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
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -f lavfi -i sine -frames:v 30 sample.mp4"
    sample = get_sample(cmd)

    demuxer = spdl.io.Demuxer(sample.path)
    decoder = spdl.io.Decoder(demuxer.video_codec)
    buffers = []
    for packets in demuxer.streaming_demux_video(30):
        print(packets)
        frames = decoder.decode(packets)
        print(frames)
        buffer = spdl.io.convert_frames(frames)
        buffers.append(spdl.io.to_numpy(buffer))
    if (frames := decoder.flush()) is not None:
        buffer = spdl.io.convert_frames(frames)
        buffers.append(spdl.io.to_numpy(buffer))

    hyp = np.concatenate(buffers)
    print(hyp.shape, hyp.dtype)

    packets = spdl.io.demux_video(sample.path)
    frames = spdl.io.decode_packets(packets)
    buffer = spdl.io.convert_frames(frames)
    ref = spdl.io.to_numpy(buffer)
    print(ref.shape, ref.dtype)

    # from PIL import Image

    # for i in [0, -1]:
    #     Image.fromarray(hyp[i]).save(f"hyp_{i}.png")
    #     Image.fromarray(ref[i]).save(f"ref_{i}.png")
    #     Image.fromarray(hyp[i] - ref[i]).save(f"diff_{i}.png")

    assert np.array_equal(hyp, ref)


def test_demuxer_get_codec():
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -f lavfi -i sine=sample_rate=48000 -af pan='stereo| c0=FR | c1=FR' -frames:v 30 sample.mp4"

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


def test_streaming_decoding_simple():
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 60 sample.mp4"

    N = 10
    sample = get_sample(cmd)

    buffer = spdl.io.load_video(sample.path)
    ref = spdl.io.to_numpy(buffer)

    demuxer = spdl.io.Demuxer(sample.path)
    decoder = spdl.io.Decoder(demuxer.video_codec)
    ite = demuxer.streaming_demux_video(N)

    buffers = []
    for i in range(6):
        packets = next(ite)
        print(f"{i}: {packets=}", flush=True)
        assert len(packets) == N
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


def test_streaming_decoding_multi():
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -f lavfi -i sine -t 15 sample.mp4"

    sample = get_sample(cmd)

    demuxer = spdl.io.Demuxer(sample.path)

    video_index = demuxer.video_stream_index
    audio_index = demuxer.audio_stream_index

    audio_decoder = spdl.io.Decoder(demuxer.audio_codec)
    video_decoder = spdl.io.Decoder(demuxer.video_codec)

    packet_stream = demuxer.streaming_demux(
        indices=[video_index, audio_index], duration=5.0
    )

    audio_buffer = []
    video_buffer = []
    for packets in packet_stream:
        if audio_index in packets:
            print(packets[audio_index])
            frames = audio_decoder.decode(packets[audio_index])
            print(frames)
            assert len(frames) > 44100 * 3
            buffer = spdl.io.convert_frames(frames)
            audio_buffer.append(spdl.io.to_numpy(buffer).T)
        if video_index in packets:
            print(packets[video_index])
            frames = video_decoder.decode(packets[video_index])
            print(frames)
            assert len(frames) > 25 * 3
            buffer = spdl.io.convert_frames(frames)
            video_buffer.append(spdl.io.to_numpy(buffer))

    if (frames := audio_decoder.flush()) is not None:
        buffer = spdl.io.convert_frames(frames)
        audio_buffer.append(spdl.io.to_numpy(buffer).T)

    if (frames := video_decoder.flush()) is not None:
        buffer = spdl.io.convert_frames(frames)
        video_buffer.append(spdl.io.to_numpy(buffer))

    hyp_audio = np.concatenate(audio_buffer).T
    hyp_video = np.concatenate(video_buffer)

    ref_audio = spdl.io.to_numpy(spdl.io.load_audio(sample.path))
    ref_video = spdl.io.to_numpy(spdl.io.load_video(sample.path))

    np.testing.assert_array_equal(hyp_audio, ref_audio)
    np.testing.assert_array_equal(hyp_video, ref_video)
