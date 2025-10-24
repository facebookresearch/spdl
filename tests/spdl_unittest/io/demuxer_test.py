# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import spdl.io

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
