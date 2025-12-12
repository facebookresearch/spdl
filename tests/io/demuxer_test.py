# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from fractions import Fraction

import spdl.io
from parameterized import parameterized

from ..fixture import FFMPEG_CLI, get_sample


class TestDemuxer(unittest.TestCase):
    def test_demuxer_query_codec(self) -> None:
        """Can fetch the codec properly."""
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc "
            "-f lavfi -i sine -t 5  sample.mp4"
        )

        sample = get_sample(cmd)

        demuxer = spdl.io.Demuxer(sample.path)

        ac = demuxer.audio_codec
        print(ac)
        self.assertEqual(ac.name, "aac")
        self.assertEqual(ac.num_channels, 1)
        self.assertEqual(ac.sample_rate, 44100)
        self.assertEqual(ac.sample_fmt, "fltp")

        vc = demuxer.video_codec
        print(vc)
        self.assertEqual(vc.name, "h264")
        self.assertEqual(vc.width, 320)
        self.assertEqual(vc.height, 240)
        self.assertEqual(vc.frame_rate, (25, 1))
        self.assertEqual(vc.pix_fmt, "yuv444p")

    def test_demuxer_query_stream_index(self) -> None:
        """Can fetch the stream index properly."""
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc "
            "-f lavfi -i sine -t 5  sample.mp4"
        )

        sample = get_sample(cmd)
        demuxer = spdl.io.Demuxer(sample.path)

        self.assertEqual(demuxer.video_stream_index, 0)
        self.assertEqual(demuxer.audio_stream_index, 1)

        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i sine "
            "-f lavfi -i testsrc -t 5 -map 0:a -map 1:v  sample.mp4"
        )

        sample = get_sample(cmd)
        demuxer = spdl.io.Demuxer(sample.path)

        self.assertEqual(demuxer.video_stream_index, 1)
        self.assertEqual(demuxer.audio_stream_index, 0)


class TestFractionalTimestamp(unittest.TestCase):
    """Test fractional timestamp/window specification."""

    def test_video_demux_with_tuple_fractions(self) -> None:
        """Can demux video using tuple fractions for timestamp."""
        # Setup: Create a 5-second test video
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc "
            "-t 5 -pix_fmt yuv444p sample.mp4"
        )
        sample = get_sample(cmd)

        # Execute: Demux video using tuple fractions (1/3 to 2/3 of video)
        demuxer = spdl.io.Demuxer(sample.path)
        packets = demuxer.demux_video(window=(Fraction(1, 3), Fraction(10, 3)))

        # Assert: Verify packets were demuxed
        self.assertIsNotNone(packets)
        self.assertGreater(len(packets), 0)
        # Verify timestamp attribute is stored as doubles
        self.assertIsNotNone(packets.timestamp)
        start, end = packets.timestamp
        self.assertAlmostEqual(start, 1.0 / 3.0, places=5)
        self.assertAlmostEqual(end, 10.0 / 3.0, places=5)

    def test_video_demux_with_fraction_class(self) -> None:
        """Can demux video using Python Fraction class for timestamp."""
        # Setup: Create a 5-second test video
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc "
            "-t 5 -pix_fmt yuv444p sample.mp4"
        )
        sample = get_sample(cmd)

        # Execute: Demux video using Fraction class
        demuxer = spdl.io.Demuxer(sample.path)
        packets = demuxer.demux_video(window=(Fraction(1, 2), Fraction(5, 2)))

        # Assert: Verify packets were demuxed
        self.assertIsNotNone(packets)
        self.assertGreater(len(packets), 0)
        # Verify timestamp is converted to doubles
        self.assertIsNotNone(packets.timestamp)
        start, end = packets.timestamp
        self.assertAlmostEqual(start, 0.5, places=5)
        self.assertAlmostEqual(end, 2.5, places=5)

    def test_video_demux_with_mixed_formats(self) -> None:
        """Can demux video using mixed timestamp formats (float and fraction)."""
        # Setup: Create a 5-second test video
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc "
            "-t 5 -pix_fmt yuv444p sample.mp4"
        )
        sample = get_sample(cmd)

        # Execute: Demux video using mixed formats (float and Fraction)
        demuxer = spdl.io.Demuxer(sample.path)
        packets = demuxer.demux_video(window=(1.0, Fraction(3, 1)))

        # Assert: Verify packets were demuxed
        self.assertIsNotNone(packets)
        self.assertGreater(len(packets), 0)
        self.assertIsNotNone(packets.timestamp)

    def test_audio_demux_with_tuple_fractions(self) -> None:
        """Can demux audio using tuple fractions for timestamp."""
        # Setup: Create a 5-second test audio
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i sine=duration=5 " "sample.mp4"
        sample = get_sample(cmd)

        # Execute: Demux audio using tuple fractions
        demuxer = spdl.io.Demuxer(sample.path)
        packets = demuxer.demux_audio(window=(Fraction(1, 2), Fraction(5, 2)))

        # Assert: Verify packets were demuxed
        self.assertIsNotNone(packets)
        self.assertGreater(len(packets), 0)
        # Verify timestamp attribute is stored as doubles
        self.assertIsNotNone(packets.timestamp)
        start, end = packets.timestamp
        self.assertAlmostEqual(start, 0.5, places=5)
        self.assertAlmostEqual(end, 2.5, places=5)

    def test_audio_demux_with_fraction_class(self) -> None:
        """Can demux audio using Python Fraction class for timestamp."""
        # Setup: Create a 5-second test audio
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i sine=duration=5 " "sample.mp4"
        sample = get_sample(cmd)

        # Execute: Demux audio using Fraction class
        demuxer = spdl.io.Demuxer(sample.path)
        packets = demuxer.demux_audio(window=(Fraction(1, 3), Fraction(2, 3)))

        # Assert: Verify packets were demuxed
        self.assertIsNotNone(packets)
        self.assertGreater(len(packets), 0)
        self.assertIsNotNone(packets.timestamp)
        start, end = packets.timestamp
        self.assertAlmostEqual(start, 1.0 / 3.0, places=5)
        self.assertAlmostEqual(end, 2.0 / 3.0, places=5)

    def test_module_level_demux_video_with_fractions(self) -> None:
        """Can use module-level demux_video function with fractions."""
        # Setup: Create a 5-second test video
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc "
            "-t 5 -pix_fmt yuv444p sample.mp4"
        )
        sample = get_sample(cmd)

        # Execute: Use module-level function with tuple fractions
        packets = spdl.io.demux_video(
            sample.path, timestamp=(Fraction(1, 4), Fraction(3, 4))
        )

        # Assert: Verify packets were demuxed
        self.assertIsNotNone(packets)
        self.assertGreater(len(packets), 0)
        self.assertIsNotNone(packets.timestamp)
        start, end = packets.timestamp
        self.assertAlmostEqual(start, 0.25, places=5)
        self.assertAlmostEqual(end, 0.75, places=5)

    def test_module_level_demux_audio_with_fractions(self) -> None:
        """Can use module-level demux_audio function with fractions."""
        # Setup: Create a 5-second test audio
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i sine=duration=5 " "sample.mp4"
        sample = get_sample(cmd)

        # Execute: Use module-level function with Fraction class
        packets = spdl.io.demux_audio(
            sample.path, timestamp=(Fraction(1, 5), Fraction(4, 5))
        )

        # Assert: Verify packets were demuxed
        self.assertIsNotNone(packets)
        self.assertGreater(len(packets), 0)
        self.assertIsNotNone(packets.timestamp)
        start, end = packets.timestamp
        self.assertAlmostEqual(start, 0.2, places=5)
        self.assertAlmostEqual(end, 0.8, places=5)

    def test_backward_compatibility_with_floats(self) -> None:
        """Backward compatibility: float timestamps still work."""
        # Setup: Create a 5-second test video
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc "
            "-t 5 -pix_fmt yuv444p sample.mp4"
        )
        sample = get_sample(cmd)

        # Execute: Demux video using traditional float timestamps
        demuxer = spdl.io.Demuxer(sample.path)
        packets = demuxer.demux_video(window=(1.0, 3.0))

        # Assert: Verify packets were demuxed
        self.assertIsNotNone(packets)
        self.assertGreater(len(packets), 0)
        self.assertIsNotNone(packets.timestamp)

    def test_precise_fraction_demuxing(self) -> None:
        """Fractions provide more precise timestamp control than floats."""
        # Setup: Create a 5-second test video
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc "
            "-t 5 -pix_fmt yuv444p sample.mp4"
        )
        sample = get_sample(cmd)

        demuxer = spdl.io.Demuxer(sample.path)

        # Execute: Demux using precise fraction (1/3)
        packets_fraction = demuxer.demux_video(window=(Fraction(1, 3), Fraction(2, 3)))

        # Execute: Demux using float approximation (0.333...)
        demuxer2 = spdl.io.Demuxer(sample.path)
        packets_float = demuxer2.demux_video(window=(0.333333333, 0.666666667))

        # Assert: Both methods produce packets, demonstrating the interface works
        # The actual timestamps might differ slightly due to precision
        self.assertIsNotNone(packets_fraction)
        self.assertGreater(len(packets_fraction), 0)
        self.assertIsNotNone(packets_float)
        self.assertGreater(len(packets_float), 0)


class TestDemuxerNameParameter(unittest.TestCase):
    """Test the name parameter for improved error messages."""

    @parameterized.expand(
        [
            ("bytes", True),
            ("file_path", False),
        ]
    )
    def test_demuxer_with_name(self, name: str, use_bytes: bool) -> None:
        """Can demux with a custom name using both bytes and file path inputs."""
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 10 sample.mp4"
        )
        sample = get_sample(cmd)

        if use_bytes:
            with open(sample.path, "rb") as f:
                src = f.read()
            custom_name = "test_video.mp4"
        else:
            src = sample.path
            custom_name = "custom_name.mp4"

        # Test with custom name
        with spdl.io.Demuxer(src, name=custom_name) as demuxer:
            packets = demuxer.demux_video()
            self.assertIsNotNone(packets)
            self.assertGreater(len(packets), 0)

    @parameterized.expand(
        [
            ("bytes", True),
            ("file_path", False),
        ]
    )
    def test_demux_audio_with_name(self, name: str, use_bytes: bool) -> None:
        """Module-level demux_audio accepts name parameter with bytes and file path."""
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i sine=frequency=1000:duration=1 sample.wav"
        sample = get_sample(cmd)

        if use_bytes:
            with open(sample.path, "rb") as f:
                src = f.read()
        else:
            src = sample.path

        packets = spdl.io.demux_audio(src, name="test_audio.wav")
        self.assertIsNotNone(packets)
        self.assertGreater(len(packets), 0)

    @parameterized.expand(
        [
            ("bytes", True),
            ("file_path", False),
        ]
    )
    def test_demux_video_with_name(self, name: str, use_bytes: bool) -> None:
        """Module-level demux_video accepts name parameter with bytes and file path."""
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 5 sample.mp4"
        sample = get_sample(cmd)

        if use_bytes:
            with open(sample.path, "rb") as f:
                src = f.read()
        else:
            src = sample.path

        packets = spdl.io.demux_video(src, name="test_video.mp4")
        self.assertIsNotNone(packets)
        self.assertGreater(len(packets), 0)

    @parameterized.expand(
        [
            ("bytes", True),
            ("file_path", False),
        ]
    )
    def test_demux_image_with_name(self, name: str, use_bytes: bool) -> None:
        """Module-level demux_image accepts name parameter with bytes and file path."""
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
        sample = get_sample(cmd)

        if use_bytes:
            with open(sample.path, "rb") as f:
                src = f.read()
        else:
            src = sample.path

        packets = spdl.io.demux_image(src, name="test_image.jpg")
        self.assertIsNotNone(packets)
        # ImagePackets doesn't have len(), just verify it's not None

    def test_name_parameter_flows_correctly(self) -> None:
        """Verify that the name parameter flows through correctly when demuxing succeeds."""
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 5 sample.mp4"
        sample = get_sample(cmd)

        with open(sample.path, "rb") as f:
            data = f.read()

        custom_name = "my_custom_video.mp4"

        # Execute: Demux should succeed with custom name
        packets = spdl.io.demux_video(data, name=custom_name)

        # Assert: Verify demux succeeded - this shows the name parameter flows through correctly
        self.assertIsNotNone(packets)
        self.assertGreater(len(packets), 0)
