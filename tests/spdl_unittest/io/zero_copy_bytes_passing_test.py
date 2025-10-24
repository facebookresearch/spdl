# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import numpy as np
import spdl.io
from parameterized import parameterized

from ..fixture import FFMPEG_CLI, get_sample

CMDS = {
    "audio": f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i sine=frequency=1000:sample_rate=48000:duration=3 -c:a pcm_s16le sample.wav",
    "video": f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4",
    "image": f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i color=0x000000,format=gray -frames:v 1 sample.png",
}


def _is_all_zero(arr):
    return all(int(v) == 0 for v in arr)


class TestDemuxBytesWithoutCopy(unittest.TestCase):
    @parameterized.expand(
        [
            ("audio",),
            ("video",),
            ("image",),
        ]
    )
    def test_demux_bytes_without_copy(self, media_type: str) -> None:
        """Data passed as bytes must be passed without copy."""
        cmd = CMDS[media_type]
        sample = get_sample(cmd)

        demux_func = {
            "audio": spdl.io.demux_audio,
            "video": spdl.io.demux_video,
            "image": spdl.io.demux_image,
        }[media_type]

        def _test(src):
            self.assertFalse(_is_all_zero(src))
            _ = demux_func(src, _zero_clear=True)
            self.assertTrue(_is_all_zero(src))

        with open(sample.path, "rb") as f:
            data = f.read()

        _test(data)


def _decode(media_type, src):
    demux_func = {
        "audio": spdl.io.demux_audio,
        "video": spdl.io.demux_video,
        "image": spdl.io.demux_image,
    }[media_type]

    packets = demux_func(src)
    frames = spdl.io.decode_packets(packets)
    buffer = spdl.io.convert_frames(frames)
    return spdl.io.to_numpy(buffer)


class TestDecodeBytes(unittest.TestCase):
    def test_decode_audio_bytes(self) -> None:
        """audio can be decoded from bytes."""
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i sine=frequency=1000:sample_rate=16000:duration=3 -c:a pcm_s16le sample.wav"
        sample = get_sample(cmd)

        ref = _decode("audio", sample.path)
        with open(sample.path, "rb") as f:
            hyp = _decode("audio", f.read())

        self.assertEqual(hyp.shape, (1, 48000))
        self.assertTrue(np.all(ref == hyp))

    def test_decode_video_bytes(self) -> None:
        """video can be decoded from bytes."""
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4"
        sample = get_sample(cmd)

        ref = _decode("video", sample.path)
        with open(sample.path, "rb") as f:
            hyp = _decode("video", f.read())

        self.assertEqual(hyp.shape, (1000, 240, 320, 3))
        self.assertTrue(np.all(ref == hyp))

    def test_demux_image_bytes(self) -> None:
        """Image (gray) can be decoded from bytes."""
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i color=0x000000,format=gray -frames:v 1 sample.png"
        sample = get_sample(cmd)

        ref = _decode("image", sample.path)
        with open(sample.path, "rb") as f:
            hyp = _decode("image", f.read())

        self.assertEqual(hyp.shape, (240, 320, 3))
        self.assertTrue(np.all(ref == hyp))
