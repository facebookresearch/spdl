# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import spdl.io

from ..fixture import FFMPEG_CLI, get_sample


def test_demuxer_query_codec():
    """Can fetch the codec properly."""
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -f lavfi -i sine -t 5  sample.mp4"

    sample = get_sample(cmd)

    demuxer = spdl.io.Demuxer(sample.path)

    ac = demuxer.audio_codec
    print(ac)
    assert ac.name == "aac"
    assert ac.num_channels == 1
    assert ac.sample_rate == 44100
    assert ac.sample_fmt == "fltp"

    vc = demuxer.video_codec
    print(vc)
    assert vc.name == "h264"
    assert vc.width == 320
    assert vc.height == 240
    assert vc.frame_rate == (25, 1)
    assert vc.pix_fmt == "yuv444p"


def test_demuxer_query_stream_index():
    """Can fetch the stream index properly."""
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -f lavfi -i sine -t 5  sample.mp4"

    sample = get_sample(cmd)
    demuxer = spdl.io.Demuxer(sample.path)

    assert demuxer.video_stream_index == 0
    assert demuxer.audio_stream_index == 1

    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i sine -f lavfi -i testsrc -t 5 -map 0:a -map 1:v  sample.mp4"

    sample = get_sample(cmd)
    demuxer = spdl.io.Demuxer(sample.path)

    assert demuxer.video_stream_index == 1
    assert demuxer.audio_stream_index == 0
