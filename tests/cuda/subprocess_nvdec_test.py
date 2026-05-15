# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing
import pickle
import unittest

import spdl.io
import spdl.io.utils
import torch

from ..fixture import FFMPEG_CLI, get_sample

if not spdl.io.utils.built_with_nvcodec():
    raise unittest.SkipTest(  # pyre-ignore: [29]
        "SPDL is not compiled with NVCODEC support"
    )


DEFAULT_CUDA = 0


def _demux_video_worker(
    path: str, queue: "multiprocessing.Queue[spdl.io.VideoPackets]"
) -> None:
    packets = spdl.io.demux_video(path)
    queue.put(packets)


class TestNvdecPickleRoundtrip(unittest.TestCase):
    def test_decode_matches_after_pickle_roundtrip(self) -> None:
        """NVDEC decode of original packets matches decode of pickled packets."""
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y "
            "-f lavfi -i testsrc=duration=1:size=64x64:rate=10 "
            "-c:v libx264 -pix_fmt yuv420p sample.mp4"
        )
        sample = get_sample(cmd)
        path = sample.path

        device_config = spdl.io.cuda_config(device_index=DEFAULT_CUDA)

        # Decode original packets
        packets = spdl.io.demux_video(path)
        ref_buffer = spdl.io.decode_packets_nvdec(packets, device_config=device_config)
        ref = spdl.io.to_torch(ref_buffer)

        # Decode after pickle roundtrip
        packets = spdl.io.demux_video(path)
        data = pickle.dumps(packets)
        restored_packets = pickle.loads(data)
        result_buffer = spdl.io.decode_packets_nvdec(
            restored_packets, device_config=device_config
        )
        result = spdl.io.to_torch(result_buffer)

        self.assertEqual(ref.shape, result.shape)
        torch.testing.assert_close(ref, result)


class TestNvdecSubprocessSerialization(unittest.TestCase):
    def test_video_demux_in_subprocess_decode_nvdec(self) -> None:
        """Demux video in subprocess, decode with NVDEC in main process."""
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y "
            "-f lavfi -i testsrc=duration=1:size=64x64:rate=10 "
            "-c:v libx264 -pix_fmt yuv420p sample.mp4"
        )
        sample = get_sample(cmd)
        path = sample.path

        device_config = spdl.io.cuda_config(device_index=DEFAULT_CUDA)

        # Decode directly for reference
        ref_packets = spdl.io.demux_video(path)
        ref_buffer = spdl.io.decode_packets_nvdec(
            ref_packets, device_config=device_config
        )
        ref = spdl.io.to_torch(ref_buffer)

        # Demux in subprocess, send via queue, decode with NVDEC in main
        ctx = multiprocessing.get_context("spawn")
        queue: multiprocessing.Queue[spdl.io.VideoPackets] = ctx.Queue()
        proc = ctx.Process(target=_demux_video_worker, args=(path, queue))
        proc.start()
        packets = queue.get(timeout=30)
        proc.join(timeout=30)

        self.assertEqual(proc.exitcode, 0)

        buffer = spdl.io.decode_packets_nvdec(packets, device_config=device_config)
        result = spdl.io.to_torch(buffer)

        self.assertEqual(ref.shape, result.shape)
        torch.testing.assert_close(ref, result)
