# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import time

import numpy as np
import pytest
import spdl.io

from ..fixture import FFMPEG_CLI, get_sample

CMDS = {
    "audio": f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=48000:duration=3' -c:a pcm_s16le sample.wav",
    "video": f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 25 sample.mp4",
    "image": f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i color=0x000000,format=gray -frames:v 1 sample.png",
}


@pytest.mark.parametrize("media_type", ["audio", "video", "image"])
def test_demux_with_codec(media_type):
    """When using demux_audio/video/image, the resulting packets contain codec"""

    cmd = CMDS[media_type]

    sample = get_sample(cmd)

    demux_method = getattr(spdl.io, f"demux_{media_type}")
    packets = demux_method(sample.path)
    codec = packets.codec
    assert codec is not None
    assert codec.name is not None

    if media_type == "video":
        packets = spdl.io.apply_bsf(packets, "null")

        codec = packets.codec
        assert codec is not None
        assert codec.name is not None


@pytest.mark.parametrize("media_type", ["video"])
def test_demux_without_codec(media_type):
    """When using streaming_demux_audio/video, the resulting packets does not contain codec"""

    cmd = CMDS[media_type]

    sample = get_sample(cmd)

    demuxer = spdl.io.Demuxer(sample.path)
    bsf = spdl.io.BSF(demuxer.video_codec, "null")
    demux_method = getattr(demuxer, f"streaming_demux_{media_type}")
    it = demux_method(num_packets=5)
    for _ in range(3):
        packets = next(it)
        codec = packets.codec
        assert codec is None

        packets = bsf.filter(packets)

        codec = packets.codec
        assert codec is None


def _load_from_packets(packets):
    frames = spdl.io.decode_packets(packets)
    buffer = spdl.io.convert_frames(frames)
    return spdl.io.to_numpy(buffer)


def test_audio_packets_attributes():
    """AudioPackets have sample_rate and num_channels attributes"""
    # fmt: off
    cmd = """
    ffmpeg -hide_banner -y \
    -f lavfi -i 'sine=sample_rate=8000:frequency=305:duration=5' \
    -f lavfi -i 'sine=sample_rate=8000:frequency=300:duration=5' \
    -filter_complex amerge  -c:a pcm_s16le sample.wav
    """
    # fmt: on
    sample = get_sample(cmd)

    packets = spdl.io.demux_audio(sample.path)
    assert packets.sample_rate == 8000
    assert packets.num_channels == 2


@pytest.mark.parametrize(
    "rate",
    [
        (30, 1),
        (60, 1),
        (120, 1),
        (30000, 1001),
    ],
)
def test_video_packets_attributes(rate):
    """VideoPackets have width, height, pixe_format attributes"""
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -r {rate[0]}/{rate[1]} -i testsrc -frames:v 25 sample.mp4"
    sample = get_sample(cmd)

    packets = spdl.io.demux_video(sample.path)
    assert packets.width == 320
    assert packets.height == 240
    assert packets.pix_fmt == "yuv444p"
    assert packets.frame_rate == rate


def test_image_packets_attributes():
    """ImagePackets have width, height, pixe_format attributes"""
    cmd = CMDS["image"]
    sample = get_sample(cmd)

    packets = spdl.io.demux_image(sample.path)
    assert packets.width == 320
    assert packets.height == 240
    assert packets.pix_fmt == "gray"


@pytest.mark.parametrize("media_type", ["audio", "video", "image"])
def test_clone_packets(media_type):
    """Cloning packets allows to decode twice"""
    cmd = CMDS[media_type]
    sample = get_sample(cmd)

    demux_func = {
        "audio": spdl.io.demux_audio,
        "video": spdl.io.demux_video,
        "image": spdl.io.demux_image,
    }[media_type]

    packets1 = demux_func(sample.path)
    packets2 = packets1.clone()

    array1 = _load_from_packets(packets1)
    array2 = _load_from_packets(packets2)

    assert np.all(array1 == array2)


@pytest.mark.parametrize("media_type", ["audio", "video", "image"])
def test_clone_invalid_packets(media_type):
    """Attempt to clone already released packet raises RuntimeError instead of segfault"""
    cmd = CMDS[media_type]
    sample = get_sample(cmd)

    demux_func = {
        "audio": spdl.io.demux_audio,
        "video": spdl.io.demux_video,
        "image": spdl.io.demux_image,
    }[media_type]

    packets = demux_func(sample.path)
    _ = spdl.io.decode_packets(packets)
    with pytest.raises(TypeError):
        packets.clone()


@pytest.mark.parametrize("media_type", ["audio", "video", "image"])
def test_clone_packets_multi(media_type):
    """Can clone multiple times"""
    cmd = CMDS[media_type]
    sample = get_sample(cmd)
    N = 100

    demux_func = {
        "audio": spdl.io.demux_audio,
        "video": spdl.io.demux_video,
        "image": spdl.io.demux_image,
    }[media_type]

    packets = demux_func(sample.path)
    clones = [packets.clone() for _ in range(N)]

    array = _load_from_packets(packets)
    arrays = [_load_from_packets(c) for c in clones]

    for i in range(N):
        assert np.all(array == arrays[i])


def test_sample_decoding_time():
    """Sample decoding works"""
    # https://stackoverflow.com/questions/63725248/how-can-i-set-gop-size-to-be-a-multiple-of-the-input-framerate
    cmd = (
        f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc "
        "-force_key_frames 'expr:eq(mod(n, 25), 0)' "
        "-frames:v 5000 sample.mp4"
    )
    # Note: You can use the following command to check that the generated video has the keyframes
    # at the expected positions:
    # Ref: https://www.reddit.com/r/ffmpeg/comments/k6su5f/how_can_i_get_an_output_of_all_keyframe/
    # Use ffprobe -loglevel error -select_streams v:0 -show_entries packet=pts_time,flags -of csv=print_section=0 sample.mp4 | grep K__
    sample = get_sample(cmd)

    indices = list(range(0, 5000, 100))

    packets = spdl.io.demux_video(sample.path)
    t0 = time.monotonic()
    frames = spdl.io.decode_packets(packets.clone())
    frames = frames[indices]
    elapsed_ref = time.monotonic() - t0
    buffer = spdl.io.convert_frames(frames)
    array_ref = spdl.io.to_numpy(buffer)

    t0 = time.monotonic()
    frames = spdl.io.sample_decode_video(packets, indices)
    elapsed = time.monotonic() - t0
    buffer = spdl.io.convert_frames(frames)
    array = spdl.io.to_numpy(buffer)

    print(f"{elapsed_ref=}, {elapsed=}")
    assert np.all(array == array_ref)

    # should be much faster than 2x
    assert elapsed_ref / 2 > elapsed


def test_sample_decoding_time_sync():
    """Sample decoding works"""
    # https://stackoverflow.com/questions/63725248/how-can-i-set-gop-size-to-be-a-multiple-of-the-input-framerate
    cmd = (
        f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc "
        "-force_key_frames 'expr:eq(mod(n, 25), 0)' "
        "-frames:v 5000 sample.mp4"
    )
    # Note: You can use the following command to check that the generated video has the keyframes
    # at the expected positions:
    # Ref: https://www.reddit.com/r/ffmpeg/comments/k6su5f/how_can_i_get_an_output_of_all_keyframe/
    # Use ffprobe -loglevel error -select_streams v:0 -show_entries packet=pts_time,flags -of csv=print_section=0 sample.mp4 | grep K__
    sample = get_sample(cmd)

    indices = list(range(0, 5000, 100))

    packets = spdl.io.demux_video(sample.path)
    t0 = time.monotonic()
    frames = spdl.io.decode_packets(packets.clone())
    frames = frames[indices]
    elapsed_ref = time.monotonic() - t0
    buffer = spdl.io.convert_frames(frames)
    array_ref = spdl.io.to_numpy(buffer)

    t0 = time.monotonic()
    frames = spdl.io.sample_decode_video(packets, indices)
    elapsed = time.monotonic() - t0
    buffer = spdl.io.convert_frames(frames)
    array = spdl.io.to_numpy(buffer)

    print(f"{elapsed_ref=}, {elapsed=}")
    assert np.all(array == array_ref)

    # should be much faster than 2x
    assert elapsed_ref / 2 > elapsed


def test_packet_len():
    """VideoPackets length should exclude the preceeding packets when timestamp is not None"""
    # 3 seconds of video with only one keyframe at the beginning.
    # Use the following command to check
    # `ffprobe -loglevel error -select_streams v:0 -show_entries packet=pts_time,flags -of csv=print_section=0 sample.mp4 | grep K__`
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -force_key_frames 'expr:eq(n, 0)' -frames:v 75 sample.mp4"
    sample = get_sample(cmd)

    ref_array = spdl.io.to_numpy(spdl.io.load_video(sample.path))

    packets = spdl.io.demux_video(sample.path, timestamp=(1.0, 2.0))
    num_packets = len(packets)

    frames = spdl.io.decode_packets(packets)
    num_frames = len(frames)
    print(f"{num_packets=}, {num_frames=}")
    assert num_packets == num_frames == 25

    array = spdl.io.to_numpy(spdl.io.convert_frames(frames))
    assert np.all(array == ref_array[25:50])


def test_sample_decoding_window():
    """sample_decode_video returns the correct frame when timestamps is specified."""
    # 10 seconds of video with only one keyframe at the beginning.
    # Use the following command to check
    # `ffprobe -loglevel error -select_streams v:0 -show_entries packet=pts_time,flags -of csv=print_section=0 sample.mp4 | grep K__`
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -force_key_frames 'expr:eq(n, 0)' -frames:v 250 sample.mp4"
    sample = get_sample(cmd)

    # 250 frames
    ref_array = spdl.io.to_numpy(spdl.io.load_video(sample.path))
    assert len(ref_array) == 250

    # frames from 25 - 50, but internally it holds 0 - 50
    packets = spdl.io.demux_video(sample.path, timestamp=(1.0, 2.0))
    assert len(packets) == 25

    # decode all to verify the pre-condition
    frames = spdl.io.decode_packets(packets.clone())
    assert len(frames) == 25
    array = spdl.io.to_numpy(spdl.io.convert_frames(frames))
    assert np.all(array == ref_array[25:50])

    # Sample decode should offset the indices
    indices = list(range(0, 25, 2))
    frames = spdl.io.sample_decode_video(packets, indices)
    assert len(indices) == len(frames) == 13
    array = spdl.io.to_numpy(spdl.io.convert_frames(frames))
    print(f"{array.shape=}, {ref_array[25:50:2].shape=}")
    assert np.all(array == ref_array[25:50:2])


def test_sample_decoding_window_sync():
    """sample_decode_video returns the correct frame when timestamps is specified."""
    # 10 seconds of video with only one keyframe at the beginning.
    # Use the following command to check
    # `ffprobe -loglevel error -select_streams v:0 -show_entries packet=pts_time,flags -of csv=print_section=0 sample.mp4 | grep K__`
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -force_key_frames 'expr:eq(n, 0)' -frames:v 250 sample.mp4"
    sample = get_sample(cmd)

    # 250 frames
    ref_array = spdl.io.to_numpy(spdl.io.load_video(sample.path))
    assert len(ref_array) == 250

    # frames from 25 - 50, but internally it holds 0 - 50
    packets = spdl.io.demux_video(sample.path, timestamp=(1.0, 2.0))
    assert len(packets) == 25

    # decode all to verify the pre-condition
    frames = spdl.io.decode_packets(packets.clone())
    assert len(frames) == 25
    array = spdl.io.to_numpy(spdl.io.convert_frames(frames))
    assert np.all(array == ref_array[25:50])

    # Sample decode should offset the indices
    indices = list(range(0, 25, 2))
    frames = spdl.io.sample_decode_video(packets, indices)
    assert len(indices) == len(frames) == 13
    array = spdl.io.to_numpy(spdl.io.convert_frames(frames))
    print(f"{array.shape=}, {ref_array[25:50:2].shape=}")
    assert np.all(array == ref_array[25:50:2])


def test_sample_decode_video_default_color_space():
    """sample_decode_video should return rgb24 frames by default."""
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 10 sample.mp4"
    sample = get_sample(cmd)

    packets = spdl.io.demux_video(sample.path)
    assert packets.pix_fmt != "rgb24"  # precondition
    frames = spdl.io.sample_decode_video(packets, list(range(10)))

    for f in frames:
        assert f.pix_fmt == "rgb24"


def test_sample_decode_video_default_color_space_sync():
    """sample_decode_video should return rgb24 frames by default."""
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 10 sample.mp4"
    sample = get_sample(cmd)

    packets = spdl.io.demux_video(sample.path)
    assert packets.pix_fmt != "rgb24"  # precondition
    frames = spdl.io.sample_decode_video(packets, list(range(10)))

    for f in frames:
        assert f.pix_fmt == "rgb24"


def test_sample_decode_video_with_windowed_packets_and_filter():
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc=rate=10 -frames:v 30 sample.mp4"
    sample = get_sample(cmd)

    timestamp = (0.55, 2.05)
    packets = spdl.io.demux_video(sample.path, timestamp=timestamp)
    filter_desc = spdl.io.get_video_filter_desc(
        scale_width=224,
        scale_height=224,
        pix_fmt="rgb24",
    )

    assert len(packets) == 15
    idx = [0, 2, 4, 6, 8, 10, 12, 14]
    frames = spdl.io.sample_decode_video(packets, idx, filter_desc=filter_desc)
    assert [f.pts for f in frames] == [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]


def test_url():
    """Demuxing from bytes reports the address."""
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc=rate=10 -frames:v 1 sample.mp4"
    sample = get_sample(cmd)

    with open(sample.path, "rb") as f:
        data = f.read()

    addr = np.frombuffer(data, dtype=np.uint8).ctypes.data
    packets = spdl.io.demux_video(data)
    assert f"Bytes: {addr:#x}" in str(packets)
