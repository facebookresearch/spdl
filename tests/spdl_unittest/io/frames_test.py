# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import spdl.io


def test_image_frame_metadata(get_sample):
    """Smoke test for image frame metadata.
    Ideally, we should use images with EXIF data, but ffmpeg
    does not seem to support exif, and I don't want to check-in
    assets data, so just smoke test.
    """
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
    sample = get_sample(cmd)

    packets = spdl.io.demux_image(sample.path)
    frames = spdl.io.decode_packets(packets)

    assert frames.metadata == {}
