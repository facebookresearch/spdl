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
    "video": f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 25 sample.mp4",
    "image": f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i color=0x000000,format=gray -frames:v 1 sample.png",
}


def _load_from_frames(frames):
    buffer = spdl.io.convert_frames(frames)
    return spdl.io.to_numpy(buffer)


class TestCloneFrames(unittest.TestCase):
    @parameterized.expand(
        [
            ("audio",),
            ("video",),
            ("image",),
        ]
    )
    def test_clone_frames(self, media_type: str) -> None:
        """Cloning frames allows to decode twice"""
        cmd = CMDS[media_type]
        sample = get_sample(cmd)

        if media_type == "audio":
            frames1 = spdl.io.decode_packets(spdl.io.demux_audio(sample.path))
        elif media_type == "video":
            frames1 = spdl.io.decode_packets(spdl.io.demux_video(sample.path))
        else:  # image
            frames1 = spdl.io.decode_packets(spdl.io.demux_image(sample.path))

        frames2 = frames1.clone()

        array1 = _load_from_frames(frames1)
        array2 = _load_from_frames(frames2)

        self.assertTrue(np.all(array1 == array2))

    @parameterized.expand(
        [
            ("audio",),
            ("video",),
            ("image",),
        ]
    )
    def test_clone_invalid_frames(self, media_type: str) -> None:
        """Attempt to clone already released frames raises RuntimeError instead of segfault"""
        cmd = CMDS[media_type]
        sample = get_sample(cmd)

        if media_type == "audio":
            frames = spdl.io.decode_packets(spdl.io.demux_audio(sample.path))
        elif media_type == "video":
            frames = spdl.io.decode_packets(spdl.io.demux_video(sample.path))
        else:  # image
            frames = spdl.io.decode_packets(spdl.io.demux_image(sample.path))

        _ = spdl.io.convert_frames(frames)
        with self.assertRaises(TypeError):
            frames.clone()

    @parameterized.expand(
        [
            ("audio",),
            ("video",),
            ("image",),
        ]
    )
    def test_clone_frames_multi(self, media_type: str) -> None:
        """Can clone multiple times"""
        cmd = CMDS[media_type]
        sample = get_sample(cmd)
        N = 100

        if media_type == "audio":
            frames = spdl.io.decode_packets(spdl.io.demux_audio(sample.path))
        elif media_type == "video":
            frames = spdl.io.decode_packets(spdl.io.demux_video(sample.path))
        else:  # image
            frames = spdl.io.decode_packets(spdl.io.demux_image(sample.path))

        clones = [frames.clone() for _ in range(N)]

        array = _load_from_frames(frames)
        arrays = [_load_from_frames(c) for c in clones]

        for i in range(N):
            self.assertTrue(np.all(array == arrays[i]))
