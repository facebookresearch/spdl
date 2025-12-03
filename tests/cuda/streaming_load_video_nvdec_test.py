# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import spdl.io
import spdl.io.utils
import torch

from ..fixture import FFMPEG_CLI, get_sample

if not spdl.io.utils.built_with_nvcodec():
    raise unittest.SkipTest("SPDL is not compiled with NVCODEC support")


DEFAULT_CUDA = 0


def _get_h264_sample():
    cmd = (
        f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i "
        "testsrc,format=yuv420p -frames:v 100 sample.mp4"
    )
    return get_sample(cmd)


class TestStreamingLoadVideoNvdec(unittest.TestCase):
    def test_streaming_load_video_nvdec_basic(self) -> None:
        """Can stream video frames using NVDEC"""
        h264 = _get_h264_sample()

        device_config = spdl.io.cuda_config(device_index=DEFAULT_CUDA)
        streamer = spdl.io.streaming_load_video_nvdec(
            h264.path,
            device_config,
            num_frames=32,
        )

        chunks = list(streamer)
        total_frames = sum(len(chunk) for chunk in chunks)

        # We expect 100 frames total (3 chunks of 32, 32, 36)
        self.assertEqual(total_frames, 100)
        self.assertEqual(len(chunks), 4)  # 32, 32, 32, 4 frames

        # Check first chunk
        first_chunk = chunks[0]
        self.assertEqual(len(first_chunk), 32)

        # Convert first buffer to tensor to validate format
        first_buffer = first_chunk[0]
        tensor = spdl.io.to_torch(first_buffer)

        # NV12 format: height = original_height + original_height // 2
        # testsrc default is 320x240, so NV12 height = 240 + 120 = 360
        # Shape is (height, width) not (channels, height, width)
        self.assertEqual(tensor.shape[0], 360)  # Height in NV12 format
        self.assertEqual(tensor.shape[1], 320)  # Width
        self.assertEqual(tensor.dtype, torch.uint8)

    def test_streaming_load_video_nvdec_with_scaling(self) -> None:
        """Can stream video frames with scaling"""
        h264 = _get_h264_sample()

        device_config = spdl.io.cuda_config(device_index=DEFAULT_CUDA)
        streamer = spdl.io.streaming_load_video_nvdec(
            h264.path,
            device_config,
            num_frames=16,
            post_processing_params={
                "scale_width": 160,
                "scale_height": 120,
            },
        )

        chunks = list(streamer)
        total_frames = sum(len(chunk) for chunk in chunks)

        self.assertEqual(total_frames, 100)

        # Check frame dimensions
        first_buffer = chunks[0][0]
        tensor = spdl.io.to_torch(first_buffer)

        # NV12 format: height = scaled_height + scaled_height // 2
        # Scaled to 160x120, so NV12 height = 120 + 60 = 180
        # Shape is (height, width) not (channels, height, width)
        self.assertEqual(tensor.shape[0], 180)  # Scaled NV12 height
        self.assertEqual(tensor.shape[1], 160)  # Scaled width

    def test_streaming_load_video_nvdec_with_cropping(self) -> None:
        """Can stream video frames with cropping"""
        h264 = _get_h264_sample()

        crop_top, crop_bottom = 40, 80
        crop_left, crop_right = 100, 50

        device_config = spdl.io.cuda_config(device_index=DEFAULT_CUDA)
        streamer = spdl.io.streaming_load_video_nvdec(
            h264.path,
            device_config,
            num_frames=25,
            post_processing_params={
                "crop_top": crop_top,
                "crop_bottom": crop_bottom,
                "crop_left": crop_left,
                "crop_right": crop_right,
            },
        )

        chunks = list(streamer)
        total_frames = sum(len(chunk) for chunk in chunks)

        self.assertEqual(total_frames, 100)

        # Check cropped dimensions
        first_buffer = chunks[0][0]
        tensor = spdl.io.to_torch(first_buffer)

        # Cropped dimensions: 320 - 100 - 50 = 170 width
        # 240 - 40 - 80 = 120 height
        # NV12 format: 120 + 60 = 180
        expected_width = 320 - crop_left - crop_right
        expected_height_yuv = 240 - crop_top - crop_bottom
        expected_nv12_height = expected_height_yuv + expected_height_yuv // 2

        # Shape is (height, width) not (channels, height, width)
        self.assertEqual(tensor.shape[0], expected_nv12_height)
        self.assertEqual(tensor.shape[1], expected_width)

    def test_streaming_load_video_nvdec_small_chunks(self) -> None:
        """Can stream video with small chunk sizes"""
        h264 = _get_h264_sample()

        device_config = spdl.io.cuda_config(device_index=DEFAULT_CUDA)
        streamer = spdl.io.streaming_load_video_nvdec(
            h264.path,
            device_config,
            num_frames=5,
        )

        chunks = list(streamer)
        total_frames = sum(len(chunk) for chunk in chunks)

        # 100 frames with chunk size of 5 should give 20 chunks
        self.assertEqual(total_frames, 100)
        self.assertEqual(len(chunks), 20)

        # Each chunk should have 5 frames
        for chunk in chunks:
            self.assertEqual(len(chunk), 5)

    def test_streaming_load_video_nvdec_to_rgb(self) -> None:
        """Can convert NV12 frames to RGB during streaming"""
        h264 = _get_h264_sample()

        device_config = spdl.io.cuda_config(device_index=DEFAULT_CUDA)
        streamer = spdl.io.streaming_load_video_nvdec(
            h264.path,
            device_config,
            num_frames=32,
        )

        # Get first chunk and convert to RGB
        first_chunk = next(iter(streamer))
        self.assertEqual(len(first_chunk), 32)

        # Convert NV12 to RGB
        rgb_buffer = spdl.io.nv12_to_rgb(first_chunk, device_config=device_config)
        rgb_tensor = spdl.io.to_torch(rgb_buffer)

        # RGB format should be (N, 3, H, W)
        self.assertEqual(rgb_tensor.shape[0], 32)  # Number of frames
        self.assertEqual(rgb_tensor.shape[1], 3)  # RGB channels
        self.assertEqual(rgb_tensor.shape[2], 240)  # Original height
        self.assertEqual(rgb_tensor.shape[3], 320)  # Original width
        self.assertEqual(rgb_tensor.dtype, torch.uint8)

    def test_streaming_load_video_nvdec_nonexistent_file(self) -> None:
        """Should raise error for non-existent file"""
        device_config = spdl.io.cuda_config(device_index=DEFAULT_CUDA)

        with self.assertRaises(RuntimeError):
            streamer = spdl.io.streaming_load_video_nvdec(
                "nonexistent_video_file.mp4",
                device_config,
                num_frames=32,
            )
            # Try to iterate - this should fail
            next(iter(streamer))

    def test_streaming_load_video_nvdec_large_chunk_size(self) -> None:
        """Can stream when chunk size is larger than total frames"""
        h264 = _get_h264_sample()

        device_config = spdl.io.cuda_config(device_index=DEFAULT_CUDA)
        streamer = spdl.io.streaming_load_video_nvdec(
            h264.path,
            device_config,
            num_frames=200,  # More than 100 frames in video
        )

        chunks = list(streamer)

        # Should get 1 chunk with all 100 frames
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 100)
