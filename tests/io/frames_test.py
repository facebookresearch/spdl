# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import spdl.io

from ..fixture import FFMPEG_CLI, get_sample


class TestFrames(unittest.TestCase):
    def test_image_frame_metadata(self) -> None:
        """Smoke test for image frame metadata.
        Ideally, we should use images with EXIF data, but ffmpeg
        does not seem to support exif, and I don't want to check-in
        assets data, so just smoke test.
        """
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
        sample = get_sample(cmd)

        packets = spdl.io.demux_image(sample.path)
        frames = spdl.io.decode_packets(packets)

        self.assertEqual(frames.metadata, {})

    def test_sample_frame_timestamps(self) -> None:
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 10 sample.mp4"
        )
        sample = get_sample(cmd)

        packets = spdl.io.demux_video(sample.path)
        frames = spdl.io.decode_packets(packets)
        print(frames.get_pts())
        print(frames.get_timestamps())
        print(frames.time_base)

        expected_pts = [0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608]
        num, den = frames.time_base

        self.assertEqual(den / num, 12800)
        self.assertEqual(frames.get_pts(), expected_pts)
        self.assertEqual(frames.get_timestamps(), [t * num / den for t in expected_pts])
