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
import spdl.utils
from spdl.io import get_audio_filter_desc, get_video_filter_desc
from spdl.lib import _libspdl


def test_failure():
    """Demuxer fails without segfault if the input does not exist"""

    with pytest.raises(RuntimeError, match="Failed to open the input"):
        spdl.io.Demuxer("dvkgviuerehidguburuekkhgjijfjbkj")


def _decode_packet(packets):
    frames = spdl.io.decode_packets(packets)
    print(frames)
    buffer = spdl.io.convert_frames(frames)
    print(buffer)
    array = spdl.io.to_numpy(buffer)
    print(array.shape, array.dtype)
    return array


def _test_decode(demux_fn, timestamps):
    # There was a case where the underlying file device was delayed, and the
    # generated sample file is not ready when the test is started, so
    # sleeping here for 1 second to make sure the file is ready.
    time.sleep(1)

    frames = []
    for timestamp in timestamps:
        packets = demux_fn(timestamp)
        print(packets)
        frames_ = _decode_packet(packets)
        frames.append(frames_)

    return frames


def test_decode_audio_clips(get_sample):
    """Can decode audio clips."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=48000:duration=3' -c:a pcm_s16le sample.wav"
    sample = get_sample(cmd)

    def _test():
        timestamps = [(i, i + 1) for i in range(2)]
        demuxer = spdl.io.Demuxer(sample.path)
        arrays = _test_decode(demuxer.demux_audio, timestamps)

        assert len(arrays) == 2
        for i, arr in enumerate(arrays):
            print(i, arr.shape, arr.dtype)
            assert arr.shape == (1, 48000)
            assert arr.dtype == np.float32

    _test()


def test_decode_audio_clips_num_frames(get_sample):
    """Can decode audio clips with padding/dropping."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=16000:duration=1' -c:a pcm_s16le sample.wav"
    sample = get_sample(cmd)

    def _decode(src, num_frames=None):
        with spdl.io.Demuxer(src) as demuxer:
            packets = demuxer.demux_audio(window=(0, 1))
            filter_desc = get_audio_filter_desc(
                timestamp=(0, 1), num_frames=num_frames, sample_fmt="s16"
            )
            frames = spdl.io.decode_packets(packets, filter_desc=filter_desc)
            buffer = spdl.io.convert_frames(frames)
            return spdl.io.to_numpy(buffer)

    def _test(src):
        arr0 = _decode(src)
        assert arr0.dtype == np.int16
        assert arr0.shape == (16000, 1)

        num_frames = 8000
        arr1 = _decode(src, num_frames=num_frames)
        assert arr1.dtype == np.int16
        assert arr1.shape == (num_frames, 1)
        assert np.all(arr1 == arr0[:num_frames])

        num_frames = 32000
        arr2 = _decode(src, num_frames=num_frames)
        assert arr2.dtype == np.int16
        assert arr2.shape == (num_frames, 1)
        assert np.all(arr2[:16000] == arr0)
        assert np.all(arr2[16000:] == 0)

    _test(sample.path)


def test_decode_video_clips(get_sample):
    """Can decode video clips."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)
    N = 10

    def _test():
        timestamps = [(i, i + 1) for i in range(N)]
        demuxer = spdl.io.Demuxer(sample.path)
        arrays = _test_decode(demuxer.demux_video, timestamps)
        assert len(arrays) == N
        for i, arr in enumerate(arrays):
            print(i, arr.shape, arr.dtype)
            assert arr.shape == (25, 240, 320, 3)
            assert arr.dtype == np.uint8

    _test()


def test_decode_video_clips_num_frames(get_sample):
    """Can decode video clips with padding/dropping."""
    if "tpad" not in spdl.utils.get_ffmpeg_filters():
        raise pytest.skip("tpad filter is not available. Install FFmepg >= 4.2.")

    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 50 sample.mp4"
    sample = get_sample(cmd)

    def _decode(src, pix_fmt="rgb24", **kwargs):
        with spdl.io.Demuxer(src) as demuxer:
            packets = demuxer.demux_video(window=(0, 2))
            filter_desc = get_video_filter_desc(
                timestamp=(0, 2), pix_fmt=pix_fmt, **kwargs
            )
            frames = spdl.io.decode_packets(packets, filter_desc=filter_desc)
            buffer = spdl.io.convert_frames(frames)
            return spdl.io.to_numpy(buffer)

    def _test(src):
        arr0 = _decode(src)
        assert arr0.dtype == np.uint8
        assert arr0.shape == (50, 240, 320, 3)

        num_frames = 25
        arr1 = _decode(src, num_frames=num_frames)
        assert arr1.dtype == np.uint8
        assert arr1.shape == (num_frames, 240, 320, 3)
        assert np.all(arr1 == arr0[:num_frames])

        num_frames = 100
        arr2 = _decode(src, num_frames=num_frames)
        assert arr2.dtype == np.uint8
        assert arr2.shape == (num_frames, 240, 320, 3)
        assert np.all(arr2[:50] == arr0)
        assert np.all(arr2[50:] == arr2[50])

        num_frames = 100
        arr2 = _decode(src, num_frames=num_frames, pad_mode="black")
        assert arr2.dtype == np.uint8
        assert arr2.shape == (num_frames, 240, 320, 3)
        assert np.all(arr2[:50] == arr0)
        assert np.all(arr2[50:] == 0)

        num_frames = 100
        arr2 = _decode(src, num_frames=num_frames, pad_mode="white")
        assert arr2.dtype == np.uint8
        assert arr2.shape == (num_frames, 240, 320, 3)
        assert np.all(arr2[:50] == arr0)
        assert np.all(arr2[50:] == 255)

    _test(sample.path)


def test_decode_video_frame_rate_pts(get_sample):
    """Applying frame rate outputs correct PTS."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -r 10 -frames:v 20 sample.mp4"
    sample = get_sample(cmd)

    def _test(src):
        packets = spdl.io.demux_video(src)
        frames_ref = spdl.io.decode_packets(packets.clone())
        frames = spdl.io.decode_packets(
            packets, filter_desc=get_video_filter_desc(frame_rate=(5, 1))
        )

        pts_ref = frames_ref._get_pts()
        pts = frames._get_pts()
        print(pts_ref, pts)

        assert np.all(pts_ref[::2] == pts)

    _test(sample.path)


def _decode_image(path):
    packets = spdl.io.demux_image(path)
    print(packets)
    frames = spdl.io.decode_packets(packets)
    print(frames)
    assert type(frames) is _libspdl.FFmpegImageFrames
    return frames


def _batch_decode_image(paths):
    return [_decode_image(path) for path in paths]


def test_decode_image(get_sample):
    """Can decode an image."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
    sample = get_sample(cmd, width=320, height=240)

    def _test(src):
        buffer = spdl.io.load_image(src)
        array = spdl.io.to_numpy(buffer)
        print(array.shape, array.dtype)
        assert array.dtype == np.uint8
        assert array.shape == (240, 320, 3)

    _test(sample.path)


def test_batch_decode_image(get_samples):
    """Can decode a batch of images."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 250 sample_%03d.jpg"
    samples = get_samples(cmd)

    flist = ["NON_EXISTING_FILE.JPG", *samples]

    def _test():
        buffer = spdl.io.load_image_batch(
            flist, width=None, height=None, pix_fmt=None, strict=False
        )
        assert buffer.__array_interface__["shape"] == (250, 3, 240, 320)

    _test()


def test_convert_audio(get_sample):
    """convert_frames can convert FFmpegAudioFrames to Buffer"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=48000:duration=3' -c:a pcm_s16le sample.wav"
    sample = get_sample(cmd)

    def _test(src):
        ts = [(1, 2)]
        demuxer = spdl.io.Demuxer(src)
        arrays = _test_decode(demuxer.demux_audio, ts)
        array = arrays[0]
        print(array.dtype, array.shape)
        assert array.shape == (1, 48000)

    _test(sample.path)


def test_convert_video(get_sample):
    """convert_frames can convert FFmpegVideoFrames to Buffer"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)

    def _test(src):
        packets = spdl.io.demux_video(src)
        frames = spdl.io.decode_packets(packets)
        buffer = spdl.io.convert_frames(frames)
        array = spdl.io.to_numpy(buffer)
        print(array.dtype, array.shape)
        assert array.shape == (1000, 240, 320, 3)

    _test(sample.path)


def test_convert_image(get_sample):
    """convert_frames can convert FFmpegImageFrames to Buffer"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
    sample = get_sample(cmd, width=320, height=240)

    def _test(src):
        frames = _decode_image(src)
        buffer = spdl.io.convert_frames(frames)
        print(buffer)
        arr = spdl.io.to_numpy(buffer)
        print(arr.dtype, arr.shape)
        assert arr.shape == (240, 320, 3)

    _test(sample.path)


def test_convert_batch_image(get_samples):
    """convert_frames can convert list[FFmpegImageFrames] to Buffer"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 4 sample_%03d.jpg"
    flist = get_samples(cmd)

    def _test(flist):
        frames = _batch_decode_image(flist)
        buffer = spdl.io.convert_frames(frames)
        print(buffer)
        arr = spdl.io.to_numpy(buffer)
        print(arr.dtype, arr.shape)
        assert arr.shape == (4, 240, 320, 3)

    _test(flist)
