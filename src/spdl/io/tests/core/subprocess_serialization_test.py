# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import multiprocessing
import unittest

import numpy as np
import spdl.io
from spdl.io.tests.fixture import FFMPEG_CLI, get_sample


def _demux_audio_worker(
    path: str, queue: "multiprocessing.Queue[spdl.io.AudioPackets]"
) -> None:
    packets = spdl.io.demux_audio(path)
    queue.put(packets)


def _demux_video_worker(
    path: str, queue: "multiprocessing.Queue[spdl.io.VideoPackets]"
) -> None:
    packets = spdl.io.demux_video(path)
    queue.put(packets)


class TestSubprocessSerialization(unittest.TestCase):
    def test_audio_demux_in_subprocess(self) -> None:
        """Demux audio in subprocess, decode in main process."""
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y "
            "-f lavfi -i sine=frequency=440:duration=1 "
            "-c:a aac sample.mp4"
        )
        sample = get_sample(cmd)
        path = sample.path

        # Decode directly for reference
        ref_packets = spdl.io.demux_audio(path)
        ref_frames = spdl.io.decode_packets(ref_packets)
        ref_buf = spdl.io.convert_frames(ref_frames)
        ref = spdl.io.to_numpy(ref_buf)

        # Demux in subprocess and decode in main
        ctx = multiprocessing.get_context("spawn")
        queue: multiprocessing.Queue[spdl.io.AudioPackets] = ctx.Queue()
        proc = ctx.Process(target=_demux_audio_worker, args=(path, queue))
        proc.start()
        packets = queue.get(timeout=30)
        proc.join(timeout=30)

        self.assertEqual(proc.exitcode, 0)

        frames = spdl.io.decode_packets(packets)
        result_buf = spdl.io.convert_frames(frames)
        result = spdl.io.to_numpy(result_buf)

        np.testing.assert_array_equal(ref, result, strict=True)

    def test_video_demux_in_subprocess(self) -> None:
        """Demux video in subprocess, decode in main process."""
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y "
            "-f lavfi -i testsrc=duration=1:size=64x64:rate=10 "
            "-c:v libx264 -pix_fmt yuv420p sample.mp4"
        )
        sample = get_sample(cmd)
        path = sample.path

        # Decode directly for reference
        ref_packets = spdl.io.demux_video(path)
        ref_frames = spdl.io.decode_packets(ref_packets)
        ref_buf = spdl.io.convert_frames(ref_frames)
        ref = spdl.io.to_numpy(ref_buf)

        # Demux in subprocess and decode in main
        ctx = multiprocessing.get_context("spawn")
        queue: multiprocessing.Queue[spdl.io.VideoPackets] = ctx.Queue()
        proc = ctx.Process(target=_demux_video_worker, args=(path, queue))
        proc.start()
        packets = queue.get(timeout=30)
        proc.join(timeout=30)

        self.assertEqual(proc.exitcode, 0)

        frames = spdl.io.decode_packets(packets)
        result_buf = spdl.io.convert_frames(frames)
        result = spdl.io.to_numpy(result_buf)

        np.testing.assert_array_equal(ref, result, strict=True)
