# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import time
import unittest

import numpy as np
import spdl.io
import spdl.io.utils
from spdl.io import get_audio_filter_desc, get_video_filter_desc

from ..fixture import FFMPEG_CLI, get_sample, get_samples


class TestDemuxer(unittest.TestCase):
    def test_failure(self) -> None:
        """Demuxer fails without segfault if the input does not exist"""

        with self.assertRaisesRegex(RuntimeError, "Failed to open the input"):
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


class TestDecodeAudio(unittest.TestCase):
    def test_decode_audio_clips(self) -> None:
        """Can decode audio clips."""
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i sine=frequency=1000:sample_rate=48000:duration=3 -c:a pcm_s16le sample.wav"
        sample = get_sample(cmd)

        def _test():
            timestamps = [(i, i + 1) for i in range(2)]
            demuxer = spdl.io.Demuxer(sample.path)
            arrays = _test_decode(demuxer.demux_audio, timestamps)

            self.assertEqual(len(arrays), 2)
            for i, arr in enumerate(arrays):
                print(i, arr.shape, arr.dtype)
                self.assertEqual(arr.shape, (1, 48000))
                self.assertEqual(arr.dtype, np.float32)

        _test()

    def test_decode_audio_clips_num_frames(self) -> None:
        """Can decode audio clips with padding/dropping."""
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i sine=frequency=1000:sample_rate=16000:duration=1 -c:a pcm_s16le sample.wav"
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
            self.assertEqual(arr0.dtype, np.int16)
            self.assertEqual(arr0.shape, (16000, 1))

            num_frames = 8000
            arr1 = _decode(src, num_frames=num_frames)
            self.assertEqual(arr1.dtype, np.int16)
            self.assertEqual(arr1.shape, (num_frames, 1))
            self.assertTrue(np.all(arr1 == arr0[:num_frames]))

            num_frames = 32000
            arr2 = _decode(src, num_frames=num_frames)
            self.assertEqual(arr2.dtype, np.int16)
            self.assertEqual(arr2.shape, (num_frames, 1))
            self.assertTrue(np.all(arr2[:16000] == arr0))
            self.assertTrue(np.all(arr2[16000:] == 0))

        _test(sample.path)

    def test_decode_audio_many_channels_6(self) -> None:
        """Can decode audio with more than 6 channels.

        See https://github.com/facebookresearch/spdl/issues/449
        """
        cmd = f"""
        {FFMPEG_CLI} \
            -y \
            -f lavfi -i sine=frequency=1000:sample_rate=48000:duration=3,aformat=sample_fmts=s16 \
            -f lavfi -i sine=frequency=1001:sample_rate=48000:duration=3,aformat=sample_fmts=s16 \
            -f lavfi -i sine=frequency=1002:sample_rate=48000:duration=3,aformat=sample_fmts=s16 \
            -f lavfi -i sine=frequency=1003:sample_rate=48000:duration=3,aformat=sample_fmts=s16 \
            -f lavfi -i sine=frequency=1004:sample_rate=48000:duration=3,aformat=sample_fmts=s16 \
            -f lavfi -i sine=frequency=1005:sample_rate=48000:duration=3,aformat=sample_fmts=s16 \
            -filter_complex [0:a][1:a][2:a][3:a][4:a][5:a]join=inputs=6:channel_layout=6.0[a] \
            -map [a] \
            sample.wav
        """
        sample = get_sample(cmd)

        buffer = spdl.io.load_audio(
            sample.path,
            filter_desc="aformat=channel_layouts=2c",
        )
        array = spdl.io.to_numpy(buffer)

        self.assertEqual(array.dtype, np.dtype("int16"))
        self.assertEqual(array.ndim, 2)
        self.assertEqual(array.shape[1], 2)

    def test_decode_audio_many_channels_7(self) -> None:
        """Can decode audio with more than 6 channels.

        See https://github.com/facebookresearch/spdl/issues/449
        """
        cmd = f"""
        {FFMPEG_CLI} \
            -y \
            -f lavfi -i sine=frequency=1000:sample_rate=48000:duration=3,aformat=sample_fmts=s16 \
            -f lavfi -i sine=frequency=1001:sample_rate=48000:duration=3,aformat=sample_fmts=s16 \
            -f lavfi -i sine=frequency=1002:sample_rate=48000:duration=3,aformat=sample_fmts=s16 \
            -f lavfi -i sine=frequency=1003:sample_rate=48000:duration=3,aformat=sample_fmts=s16 \
            -f lavfi -i sine=frequency=1004:sample_rate=48000:duration=3,aformat=sample_fmts=s16 \
            -f lavfi -i sine=frequency=1005:sample_rate=48000:duration=3,aformat=sample_fmts=s16 \
            -f lavfi -i sine=frequency=1006:sample_rate=48000:duration=3,aformat=sample_fmts=s16 \
            -filter_complex [0:a][1:a][2:a][3:a][4:a][5:a][6:a]join=inputs=7:channel_layout=7.0[a] \
            -map [a] \
            sample.wav
        """
        sample = get_sample(cmd)

        buffer = spdl.io.load_audio(
            sample.path,
            filter_desc="aformat=channel_layouts=2c",
        )
        array = spdl.io.to_numpy(buffer)

        self.assertEqual(array.dtype, np.dtype("int16"))
        self.assertEqual(array.ndim, 2)
        self.assertEqual(array.shape[1], 2)


class TestDecodeVideo(unittest.TestCase):
    def test_decode_video_clips(self) -> None:
        """Can decode video clips."""
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4"
        sample = get_sample(cmd)
        N = 10

        def _test():
            timestamps = [(i, i + 1) for i in range(N)]
            demuxer = spdl.io.Demuxer(sample.path)
            arrays = _test_decode(demuxer.demux_video, timestamps)
            self.assertEqual(len(arrays), N)
            for i, arr in enumerate(arrays):
                print(i, arr.shape, arr.dtype)
                self.assertEqual(arr.shape, (25, 240, 320, 3))
                self.assertEqual(arr.dtype, np.uint8)

        _test()

    # This test requires "tpad filter (FFmepg >= 4.2.")
    @unittest.skipUnless(
        "tpad" in spdl.io.utils.get_ffmpeg_filters(), "tpad filter not available"
    )
    def test_batch_video_conversion(self) -> None:
        """Can decode video clips."""
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4"
        sample = get_sample(cmd)

        timestamps = [(0, 1), (1, 1.5), (2, 2.7), (3, 3.6)]

        demuxer = spdl.io.Demuxer(sample.path)
        frames = []
        for ts in timestamps:
            packets = demuxer.demux_video(timestamp=ts)
            filter_desc = spdl.io.get_filter_desc(packets, num_frames=15)
            frames_ = spdl.io.decode_packets(packets, filter_desc=filter_desc)
            frames.append(frames_)

        buffer = spdl.io.convert_frames(frames)
        array = spdl.io.to_numpy(buffer)
        self.assertEqual(array.shape, (4, 15, 3, 240, 320))

    @unittest.skipUnless(
        "tpad" in spdl.io.utils.get_ffmpeg_filters(), "tpad filter not available"
    )
    def test_decode_video_clips_num_frames(self) -> None:
        """Can decode video clips with padding/dropping."""
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 50 sample.mp4"
        )
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
            self.assertEqual(arr0.dtype, np.uint8)
            self.assertEqual(arr0.shape, (50, 240, 320, 3))

            num_frames = 25
            arr1 = _decode(src, num_frames=num_frames)
            self.assertEqual(arr1.dtype, np.uint8)
            self.assertEqual(arr1.shape, (num_frames, 240, 320, 3))
            self.assertTrue(np.all(arr1 == arr0[:num_frames]))

            num_frames = 100
            arr2 = _decode(src, num_frames=num_frames)
            self.assertEqual(arr2.dtype, np.uint8)
            self.assertEqual(arr2.shape, (num_frames, 240, 320, 3))
            self.assertTrue(np.all(arr2[:50] == arr0))
            self.assertTrue(np.all(arr2[50:] == arr2[50]))

            num_frames = 100
            arr2 = _decode(src, num_frames=num_frames, pad_mode="black")
            self.assertEqual(arr2.dtype, np.uint8)
            self.assertEqual(arr2.shape, (num_frames, 240, 320, 3))
            self.assertTrue(np.all(arr2[:50] == arr0))
            self.assertTrue(np.all(arr2[50:] == 0))

            num_frames = 100
            arr2 = _decode(src, num_frames=num_frames, pad_mode="white")
            self.assertEqual(arr2.dtype, np.uint8)
            self.assertEqual(arr2.shape, (num_frames, 240, 320, 3))
            self.assertTrue(np.all(arr2[:50] == arr0))
            self.assertTrue(np.all(arr2[50:] == 255))

        _test(sample.path)

    def test_decode_video_frame_rate_pts(self) -> None:
        """Applying frame rate outputs correct PTS."""
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -r 10 -frames:v 20 sample.mp4"
        sample = get_sample(cmd)

        def _test(src):
            packets = spdl.io.demux_video(src)
            frames_ref = spdl.io.decode_packets(packets.clone())
            frames = spdl.io.decode_packets(
                packets, filter_desc=get_video_filter_desc(frame_rate=(5, 1))
            )

            pts_ref = frames_ref.get_timestamps()
            pts = frames.get_timestamps()
            print(pts_ref, pts)

            self.assertTrue(np.all(pts_ref[::2] == pts))

        _test(sample.path)


class TestConvertFrames(unittest.TestCase):
    def test_convert_audio(self) -> None:
        """convert_frames can convert AudioFrames to Buffer"""
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i sine=frequency=1000:sample_rate=48000:duration=3 -c:a pcm_s16le sample.wav"
        sample = get_sample(cmd)

        def _test(src):
            ts = [(1, 2)]
            demuxer = spdl.io.Demuxer(src)
            arrays = _test_decode(demuxer.demux_audio, ts)
            array = arrays[0]
            print(array.dtype, array.shape)
            self.assertEqual(array.shape, (1, 48000))

        _test(sample.path)

    def test_convert_video(self) -> None:
        """convert_frames can convert VideoFrames to Buffer"""
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4"
        sample = get_sample(cmd)

        def _test(src):
            packets = spdl.io.demux_video(src)
            frames = spdl.io.decode_packets(packets)
            buffer = spdl.io.convert_frames(frames)
            array = spdl.io.to_numpy(buffer)
            print(array.dtype, array.shape)
            self.assertEqual(array.shape, (1000, 240, 320, 3))

        _test(sample.path)

    def test_convert_image(self) -> None:
        """convert_frames can convert ImageFrames to Buffer"""
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
        sample = get_sample(cmd)

        frames = spdl.io.decode_packets(spdl.io.demux_image(sample.path))
        buffer = spdl.io.convert_frames(frames)
        print(buffer)
        arr = spdl.io.to_numpy(buffer)
        print(arr.dtype, arr.shape)
        self.assertEqual(arr.shape, (240, 320, 3))

    def test_convert_batch_image(self) -> None:
        """convert_frames can convert list[ImageFrames] to Buffer"""
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 4 sample_%03d.jpg"
        srcs = get_samples(cmd)

        frames = [spdl.io.decode_packets(spdl.io.demux_image(s.path)) for s in srcs]
        buffer = spdl.io.convert_frames(frames)
        print(buffer)
        arr = spdl.io.to_numpy(buffer)
        print(arr.dtype, arr.shape)
        self.assertEqual(arr.shape, (4, 240, 320, 3))
