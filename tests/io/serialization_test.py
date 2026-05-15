# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pickle
import unittest

import numpy as np
import spdl.io

from ..fixture import FFMPEG_CLI, get_sample


class TestAudioPacketsSerialization(unittest.TestCase):
    def test_pickle_roundtrip(self) -> None:
        """AudioPackets can be pickled and unpickled."""
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y "
            "-f lavfi -i sine=frequency=440:duration=1 "
            "-c:a aac sample.mp4"
        )
        sample = get_sample(cmd)
        packets = spdl.io.demux_audio(sample.path)

        data = pickle.dumps(packets)
        restored = pickle.loads(data)

        self.assertEqual(len(packets), len(restored))
        self.assertEqual(packets.sample_rate, restored.sample_rate)
        self.assertEqual(packets.num_channels, restored.num_channels)
        self.assertEqual(packets.timestamp, restored.timestamp)

    def test_decode_after_unpickle(self) -> None:
        """Decoding unpickled AudioPackets produces identical output."""
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y "
            "-f lavfi -i sine=frequency=440:duration=1 "
            "-c:a aac sample.mp4"
        )
        sample = get_sample(cmd)
        packets = spdl.io.demux_audio(sample.path)

        original_frames = spdl.io.decode_packets(packets.clone())
        original_buf = spdl.io.convert_frames(original_frames)
        original = spdl.io.to_numpy(original_buf)

        restored = pickle.loads(pickle.dumps(packets))
        restored_frames = spdl.io.decode_packets(restored)
        restored_buf = spdl.io.convert_frames(restored_frames)
        result = spdl.io.to_numpy(restored_buf)

        np.testing.assert_array_equal(original, result, strict=True)

    def test_side_data_preserved(self) -> None:
        """AAC in MP4 has SKIP_SAMPLES side data; decoding after pickle is correct."""
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y "
            "-f lavfi -i sine=frequency=440:duration=1 "
            "-c:a aac sample.mp4"
        )
        sample = get_sample(cmd)
        packets = spdl.io.demux_audio(sample.path)

        frames = spdl.io.decode_packets(packets.clone())
        ref_buf = spdl.io.convert_frames(frames)
        ref = spdl.io.to_numpy(ref_buf)

        restored = pickle.loads(pickle.dumps(packets))
        frames2 = spdl.io.decode_packets(restored)
        result_buf = spdl.io.convert_frames(frames2)
        result = spdl.io.to_numpy(result_buf)

        np.testing.assert_array_equal(ref, result, strict=True)


class TestVideoPacketsSerialization(unittest.TestCase):
    def test_pickle_roundtrip(self) -> None:
        """VideoPackets can be pickled and unpickled."""
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y "
            "-f lavfi -i testsrc=duration=1:size=64x64:rate=10 "
            "-c:v libx264 -pix_fmt yuv420p sample.mp4"
        )
        sample = get_sample(cmd)
        packets = spdl.io.demux_video(sample.path)

        data = pickle.dumps(packets)
        restored = pickle.loads(data)

        self.assertEqual(len(packets), len(restored))
        self.assertEqual(packets.width, restored.width)
        self.assertEqual(packets.height, restored.height)
        self.assertEqual(packets.pix_fmt, restored.pix_fmt)
        self.assertEqual(packets.frame_rate, restored.frame_rate)
        self.assertEqual(packets.timestamp, restored.timestamp)

    def test_decode_after_unpickle(self) -> None:
        """Decoding unpickled VideoPackets produces identical output."""
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y "
            "-f lavfi -i testsrc=duration=1:size=64x64:rate=10 "
            "-c:v libx264 -pix_fmt yuv420p sample.mp4"
        )
        sample = get_sample(cmd)
        packets = spdl.io.demux_video(sample.path)

        original_frames = spdl.io.decode_packets(packets.clone())
        original_buf = spdl.io.convert_frames(original_frames)
        original = spdl.io.to_numpy(original_buf)

        restored = pickle.loads(pickle.dumps(packets))
        restored_frames = spdl.io.decode_packets(restored)
        restored_buf = spdl.io.convert_frames(restored_frames)
        result = spdl.io.to_numpy(restored_buf)

        np.testing.assert_array_equal(original, result, strict=True)

    def test_multiple_frames(self) -> None:
        """Serialization preserves all frames; decode results match."""
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y "
            "-f lavfi -i testsrc=duration=1:size=64x64:rate=10 "
            "-c:v libx264 -pix_fmt yuv420p sample.mp4"
        )
        sample = get_sample(cmd)
        packets = spdl.io.demux_video(sample.path)

        self.assertGreater(len(packets), 1)

        original_frames = spdl.io.decode_packets(packets.clone())
        original_buf = spdl.io.convert_frames(original_frames)
        original = spdl.io.to_numpy(original_buf)

        restored = pickle.loads(pickle.dumps(packets))
        self.assertEqual(len(packets), len(restored))

        ts_orig = packets.get_timestamps(raw=True)
        ts_restored = restored.get_timestamps(raw=True)
        self.assertEqual(ts_orig, ts_restored)

        restored_frames = spdl.io.decode_packets(restored)
        restored_buf = spdl.io.convert_frames(restored_frames)
        result = spdl.io.to_numpy(restored_buf)

        np.testing.assert_array_equal(original, result, strict=True)

    def test_with_timestamp_window(self) -> None:
        """Serialization preserves timestamp window; decode results match."""
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y "
            "-f lavfi -i testsrc=duration=2:size=64x64:rate=10 "
            "-c:v libx264 -pix_fmt yuv420p sample.mp4"
        )
        sample = get_sample(cmd)
        packets = spdl.io.demux_video(sample.path, timestamp=(0.5, 1.5))

        original_frames = spdl.io.decode_packets(packets.clone())
        original_buf = spdl.io.convert_frames(original_frames)
        original = spdl.io.to_numpy(original_buf)

        restored = pickle.loads(pickle.dumps(packets))
        self.assertEqual(packets.timestamp, restored.timestamp)

        restored_frames = spdl.io.decode_packets(restored)
        restored_buf = spdl.io.convert_frames(restored_frames)
        result = spdl.io.to_numpy(restored_buf)

        np.testing.assert_array_equal(original, result, strict=True)


class TestImagePacketsSerialization(unittest.TestCase):
    def test_pickle_roundtrip(self) -> None:
        """ImagePackets can be pickled and unpickled."""
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y "
            "-f lavfi -i testsrc=size=64x64 -frames:v 1 sample.jpg"
        )
        sample = get_sample(cmd)
        packets = spdl.io.demux_image(sample.path)

        data = pickle.dumps(packets)
        restored = pickle.loads(data)

        self.assertEqual(packets.width, restored.width)
        self.assertEqual(packets.height, restored.height)
        self.assertEqual(packets.pix_fmt, restored.pix_fmt)

    def test_decode_after_unpickle(self) -> None:
        """Decoding unpickled ImagePackets produces identical output."""
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y "
            "-f lavfi -i testsrc=size=64x64 -frames:v 1 sample.png"
        )
        sample = get_sample(cmd)
        packets = spdl.io.demux_image(sample.path)

        original_frames = spdl.io.decode_packets(packets.clone())
        original_buf = spdl.io.convert_frames(original_frames)
        original = spdl.io.to_numpy(original_buf)

        restored = pickle.loads(pickle.dumps(packets))
        restored_frames = spdl.io.decode_packets(restored)
        restored_buf = spdl.io.convert_frames(restored_frames)
        result = spdl.io.to_numpy(restored_buf)

        np.testing.assert_array_equal(original, result, strict=True)
