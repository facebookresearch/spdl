# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import time
import unittest
from typing import cast

import numpy as np
import spdl.io
from parameterized import parameterized

from ..fixture import FFMPEG_CLI, get_sample

CMDS = {
    "audio": f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i sine=frequency=1000:sample_rate=48000:duration=3 -c:a pcm_s16le sample.wav",
    "video": f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 25 sample.mp4",
    "image": f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i color=0x000000,format=gray -frames:v 1 sample.png",
}


class TestDemuxWithCodec(unittest.TestCase):
    @parameterized.expand(
        [
            ("audio",),
            ("video",),
            ("image",),
        ]
    )
    def test_demux_with_codec(self, media_type: str) -> None:
        """When using demux_audio/video/image, the resulting packets contain codec"""

        cmd = CMDS[media_type]

        sample = get_sample(cmd)

        demux_method = getattr(spdl.io, f"demux_{media_type}")
        packets = demux_method(sample.path)
        codec = packets.codec
        self.assertIsNotNone(codec)
        self.assertIsNotNone(codec.name)

        if media_type == "video":
            packets = spdl.io.apply_bsf(packets, "null")

            codec = packets.codec
            self.assertIsNotNone(codec)
            self.assertIsNotNone(codec.name)


class TestDemuxWithoutCodec(unittest.TestCase):
    def test_demux_without_codec(self) -> None:
        """When using streaming_demux, the resulting packets does not contain codec"""

        cmd = f'{FFMPEG_CLI} -lavfi "testsrc;sine" -t 10 out.mp4'

        sample = get_sample(cmd)

        demuxer = spdl.io.Demuxer(sample.path)
        v_bsf = spdl.io.BSF(demuxer.video_codec, "null")
        a_bsf = spdl.io.BSF(demuxer.audio_codec, "null")
        num_packets = 0
        for packets in demuxer.streaming_demux(num_packets=5):
            num_packets += 1
            codec = packets.codec
            self.assertIsNone(codec)

            if packets.__class__.__name__ == "AudioPackets":
                packets = a_bsf.filter(packets)
            elif packets.__class__.__name__ == "VideoPackets":
                packets = v_bsf.filter(packets)
            else:
                raise RuntimeError("Unexpected packet type")

            codec = packets.codec
            self.assertIsNone(codec)
        self.assertGreater(num_packets, 10)


def _load_from_packets(packets):
    frames = spdl.io.decode_packets(packets)
    buffer = spdl.io.convert_frames(frames)
    return spdl.io.to_numpy(buffer)


class TestAudioPacketsAttributes(unittest.TestCase):
    def test_audio_packets_attributes(self) -> None:
        """AudioPackets have sample_rate and num_channels attributes"""
        # fmt: off
        cmd = f"""
        {FFMPEG_CLI} -hide_banner -y \
        -f lavfi -i sine=sample_rate=8000:frequency=305:duration=5 \
        -f lavfi -i sine=sample_rate=8000:frequency=300:duration=5 \
        -filter_complex amerge  -c:a pcm_s16le sample.wav
        """
        # fmt: on
        sample = get_sample(cmd)

        packets = spdl.io.demux_audio(sample.path)
        self.assertEqual(packets.sample_rate, 8000)
        self.assertEqual(packets.num_channels, 2)


class TestVideoPacketsAttributes(unittest.TestCase):
    @parameterized.expand(
        [
            ((30, 1),),
            ((60, 1),),
            ((120, 1),),
            ((30000, 1001),),
        ]
    )
    def test_video_packets_attributes(self, rate: tuple[int, int]) -> None:
        """VideoPackets have width, height, pixe_format attributes"""
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -r {rate[0]}/{rate[1]} -i testsrc -frames:v 25 sample.mp4"
        sample = get_sample(cmd)

        packets = spdl.io.demux_video(sample.path)
        self.assertEqual(packets.width, 320)
        self.assertEqual(packets.height, 240)
        self.assertEqual(packets.pix_fmt, "yuv444p")
        self.assertEqual(packets.frame_rate, rate)


class TestImagePacketsAttributes(unittest.TestCase):
    def test_image_packets_attributes(self) -> None:
        """ImagePackets have width, height, pixe_format attributes"""
        cmd = CMDS["image"]
        sample = get_sample(cmd)

        packets = spdl.io.demux_image(sample.path)
        self.assertEqual(packets.width, 320)
        self.assertEqual(packets.height, 240)
        self.assertEqual(packets.pix_fmt, "gray")


class TestClonePackets(unittest.TestCase):
    @parameterized.expand(
        [
            ("audio",),
            ("video",),
            ("image",),
        ]
    )
    def test_clone_packets(self, media_type: str) -> None:
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

        self.assertTrue(np.all(array1 == array2))

    @parameterized.expand(
        [
            ("audio",),
            ("video",),
            ("image",),
        ]
    )
    def test_clone_invalid_packets(self, media_type: str) -> None:
        """Attempt to clone already released packet raises RuntimeError instead of segfault"""
        cmd = CMDS[media_type]
        sample = get_sample(cmd)

        if media_type == "audio":
            packets = spdl.io.demux_audio(sample.path)
            _ = spdl.io.decode_packets(packets)
        elif media_type == "video":
            packets = spdl.io.demux_video(sample.path)
            _ = spdl.io.decode_packets(packets)
        elif media_type == "image":
            packets = spdl.io.demux_image(sample.path)
            _ = spdl.io.decode_packets(packets)
        else:
            raise RuntimeError("Unexpected media type")

        with self.assertRaises(TypeError):
            packets.clone()

    @parameterized.expand(
        [
            ("audio",),
            ("video",),
            ("image",),
        ]
    )
    def test_clone_packets_multi(self, media_type: str) -> None:
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
            self.assertTrue(np.all(array == arrays[i]))


class TestSampleDecodingTime(unittest.TestCase):
    def test_sample_decoding_time(self) -> None:
        """Sample decoding works"""
        # https://stackoverflow.com/questions/63725248/how-can-i-set-gop-size-to-be-a-multiple-of-the-input-framerate
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc "
            '-force_key_frames "expr:eq(mod(n, 25), 0)" '
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
        frames = cast(
            list[spdl.io.ImageFrames], spdl.io.sample_decode_video(packets, indices)
        )
        elapsed = time.monotonic() - t0
        buffer = spdl.io.convert_frames(frames)
        array = spdl.io.to_numpy(buffer)

        print(f"{elapsed_ref=}, {elapsed=}")
        self.assertTrue(np.all(array == array_ref))

        # should be much faster than 2x
        self.assertGreater(elapsed_ref / 2, elapsed)

    def test_sample_decoding_time_sync(self) -> None:
        """Sample decoding works"""
        # https://stackoverflow.com/questions/63725248/how-can-i-set-gop-size-to-be-a-multiple-of-the-input-framerate
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc "
            '-force_key_frames "expr:eq(mod(n, 25), 0)" '
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
        frames = cast(
            list[spdl.io.ImageFrames], spdl.io.sample_decode_video(packets, indices)
        )
        elapsed = time.monotonic() - t0
        buffer = spdl.io.convert_frames(frames)
        array = spdl.io.to_numpy(buffer)

        print(f"{elapsed_ref=}, {elapsed=}")
        self.assertTrue(np.all(array == array_ref))

        # should be much faster than 2x
        self.assertGreater(elapsed_ref / 2, elapsed)


class TestPacketLen(unittest.TestCase):
    def test_packet_len(self) -> None:
        """VideoPackets length should exclude the preceding packets when timestamp is not None"""
        # 3 seconds of video with only one keyframe at the beginning.
        # Use the following command to check
        # `ffprobe -loglevel error -select_streams v:0 -show_entries packet=pts_time,flags -of csv=print_section=0 sample.mp4 | grep K__`
        cmd = f'{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -force_key_frames "expr:eq(n, 0)" -frames:v 75 sample.mp4'
        sample = get_sample(cmd)

        ref_array = spdl.io.to_numpy(spdl.io.load_video(sample.path))

        packets = spdl.io.demux_video(sample.path, timestamp=(1.0, 2.0))
        num_packets = len(packets)

        frames = spdl.io.decode_packets(packets)
        num_frames = len(frames)
        print(f"{num_packets=}, {num_frames=}")
        self.assertEqual(num_packets, 25)
        self.assertEqual(num_frames, 25)

        array = spdl.io.to_numpy(spdl.io.convert_frames(frames))
        self.assertTrue(np.all(array == ref_array[25:50]))


class TestSampleDecodingWindow(unittest.TestCase):
    def test_sample_decoding_window(self) -> None:
        """sample_decode_video returns the correct frame when timestamps is specified."""
        # 10 seconds of video with only one keyframe at the beginning.
        # Use the following command to check
        # `ffprobe -loglevel error -select_streams v:0 -show_entries packet=pts_time,flags -of csv=print_section=0 sample.mp4 | grep K__`
        cmd = f'{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -force_key_frames "expr:eq(n, 0)" -frames:v 250 sample.mp4'
        sample = get_sample(cmd)

        # 250 frames
        ref_array = spdl.io.to_numpy(spdl.io.load_video(sample.path))
        self.assertEqual(len(ref_array), 250)

        # frames from 25 - 50, but internally it holds 0 - 50
        packets = spdl.io.demux_video(sample.path, timestamp=(1.0, 2.0))
        self.assertEqual(len(packets), 25)

        # decode all to verify the pre-condition
        frames = spdl.io.decode_packets(packets.clone())
        self.assertEqual(len(frames), 25)
        array = spdl.io.to_numpy(spdl.io.convert_frames(frames))
        self.assertTrue(np.all(array == ref_array[25:50]))

        # Sample decode should offset the indices
        indices = list(range(0, 25, 2))
        frames = cast(
            list[spdl.io.ImageFrames], spdl.io.sample_decode_video(packets, indices)
        )
        self.assertEqual(len(indices), 13)
        self.assertEqual(len(frames), 13)
        array = spdl.io.to_numpy(spdl.io.convert_frames(frames))
        print(f"{array.shape=}, {ref_array[25:50:2].shape=}")
        self.assertTrue(np.all(array == ref_array[25:50:2]))

    def test_sample_decoding_window_sync(self) -> None:
        """sample_decode_video returns the correct frame when timestamps is specified."""
        # 10 seconds of video with only one keyframe at the beginning.
        # Use the following command to check
        # `ffprobe -loglevel error -select_streams v:0 -show_entries packet=pts_time,flags -of csv=print_section=0 sample.mp4 | grep K__`
        cmd = f'{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -force_key_frames "expr:eq(n, 0)" -frames:v 250 sample.mp4'
        sample = get_sample(cmd)

        # 250 frames
        ref_array = spdl.io.to_numpy(spdl.io.load_video(sample.path))
        self.assertEqual(len(ref_array), 250)

        # frames from 25 - 50, but internally it holds 0 - 50
        packets = spdl.io.demux_video(sample.path, timestamp=(1.0, 2.0))
        self.assertEqual(len(packets), 25)

        # decode all to verify the pre-condition
        frames = spdl.io.decode_packets(packets.clone())
        self.assertEqual(len(frames), 25)
        array = spdl.io.to_numpy(spdl.io.convert_frames(frames))
        self.assertTrue(np.all(array == ref_array[25:50]))

        # Sample decode should offset the indices
        indices = list(range(0, 25, 2))
        frames = cast(
            list[spdl.io.ImageFrames], spdl.io.sample_decode_video(packets, indices)
        )
        self.assertEqual(len(indices), 13)
        self.assertEqual(len(frames), 13)
        array = spdl.io.to_numpy(spdl.io.convert_frames(frames))
        print(f"{array.shape=}, {ref_array[25:50:2].shape=}")
        self.assertTrue(np.all(array == ref_array[25:50:2]))


class TestSampleDecodeVideoDefaultColorSpace(unittest.TestCase):
    def test_sample_decode_video_default_color_space(self) -> None:
        """sample_decode_video should return rgb24 frames by default."""
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 10 sample.mp4"
        )
        sample = get_sample(cmd)

        packets = spdl.io.demux_video(sample.path)
        self.assertNotEqual(packets.pix_fmt, "rgb24")  # precondition
        frames = spdl.io.sample_decode_video(packets, list(range(10)))

        for f in frames:
            self.assertEqual(f.pix_fmt, "rgb24")

    def test_sample_decode_video_default_color_space_sync(self) -> None:
        """sample_decode_video should return rgb24 frames by default."""
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 10 sample.mp4"
        )
        sample = get_sample(cmd)

        packets = spdl.io.demux_video(sample.path)
        self.assertNotEqual(packets.pix_fmt, "rgb24")  # precondition
        frames = spdl.io.sample_decode_video(packets, list(range(10)))

        for f in frames:
            self.assertEqual(f.pix_fmt, "rgb24")


class TestSampleDecodeVideoWithWindowedPacketsAndFilter(unittest.TestCase):
    def test_sample_decode_video_with_windowed_packets_and_filter(self) -> None:
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc=rate=10 -frames:v 30 sample.mp4"
        sample = get_sample(cmd)

        timestamp = (0.55, 2.05)
        packets = spdl.io.demux_video(sample.path, timestamp=timestamp)
        filter_desc = spdl.io.get_video_filter_desc(
            scale_width=224,
            scale_height=224,
            pix_fmt="rgb24",
        )

        self.assertEqual(len(packets), 15)
        idx = [0, 2, 4, 6, 8, 10, 12, 14]
        frames = cast(
            list[spdl.io.ImageFrames],
            spdl.io.sample_decode_video(packets, idx, filter_desc=filter_desc),
        )
        self.assertEqual(
            [f.pts for f in frames],
            [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        )


class TestSamplePacketGetTimestamps(unittest.TestCase):
    def test_sample_packet_get_timestamps(self) -> None:
        """The timestamp of each packet can be obtained with get_timestamps method"""
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 10 sample.mp4"
        )
        sample = get_sample(cmd)

        packets = spdl.io.demux_video(sample.path)
        ts = packets.get_timestamps()
        self.assertTrue(np.array_equal(ts, [t / 25 for t in range(10)]))


class TestUrl(unittest.TestCase):
    def test_url(self) -> None:
        """Demuxing from bytes reports the address."""
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc=rate=10 -frames:v 1 sample.mp4"
        sample = get_sample(cmd)

        with open(sample.path, "rb") as f:
            data = f.read()

        addr = np.frombuffer(data, dtype=np.uint8).ctypes.data
        packets = spdl.io.demux_video(data)
        self.assertIn(f"Bytes: {addr:#x}", str(packets))
