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

        batches = list(streamer)

        # Calculate total frames from all batches using CUDA array interface
        total_frames = sum(
            batch.__cuda_array_interface__["shape"][0] for batch in batches
        )

        # We expect 100 frames total (3 batches of 32, 32, 32, 4)
        self.assertEqual(total_frames, 100)
        self.assertEqual(len(batches), 4)  # 32, 32, 32, 4 frames

        # Check first batch
        first_batch = batches[0]
        batch_shape = first_batch.__cuda_array_interface__["shape"]
        self.assertEqual(batch_shape[0], 32)  # 32 frames in batch

        # Convert first batch to tensor to validate format
        tensor = spdl.io.to_torch(first_batch)

        # NV12 format: height = original_height + original_height // 2
        # testsrc default is 320x240, so NV12 height = 240 + 120 = 360
        # Shape is (num_frames, height, width)
        self.assertEqual(tensor.shape[0], 32)  # Number of frames
        self.assertEqual(tensor.shape[1], 360)  # Height in NV12 format
        self.assertEqual(tensor.shape[2], 320)  # Width
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

        batches = list(streamer)
        total_frames = sum(
            batch.__cuda_array_interface__["shape"][0] for batch in batches
        )

        self.assertEqual(total_frames, 100)

        # Check frame dimensions
        first_batch = batches[0]
        tensor = spdl.io.to_torch(first_batch)

        # NV12 format: height = scaled_height + scaled_height // 2
        # Scaled to 160x120, so NV12 height = 120 + 60 = 180
        # Shape is (num_frames, height, width)
        self.assertEqual(tensor.shape[0], 16)  # Number of frames
        self.assertEqual(tensor.shape[1], 180)  # Scaled NV12 height
        self.assertEqual(tensor.shape[2], 160)  # Scaled width

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

        batches = list(streamer)
        total_frames = sum(
            batch.__cuda_array_interface__["shape"][0] for batch in batches
        )

        self.assertEqual(total_frames, 100)

        # Check cropped dimensions
        first_batch = batches[0]
        tensor = spdl.io.to_torch(first_batch)

        # Cropped dimensions: 320 - 100 - 50 = 170 width
        # 240 - 40 - 80 = 120 height
        # NV12 format: 120 + 60 = 180
        expected_width = 320 - crop_left - crop_right
        expected_height_yuv = 240 - crop_top - crop_bottom
        expected_nv12_height = expected_height_yuv + expected_height_yuv // 2

        # Shape is (num_frames, height, width)
        self.assertEqual(tensor.shape[0], 25)  # Number of frames
        self.assertEqual(tensor.shape[1], expected_nv12_height)
        self.assertEqual(tensor.shape[2], expected_width)

    def test_streaming_load_video_nvdec_small_chunks(self) -> None:
        """Can stream video with small chunk sizes"""
        h264 = _get_h264_sample()

        device_config = spdl.io.cuda_config(device_index=DEFAULT_CUDA)
        streamer = spdl.io.streaming_load_video_nvdec(
            h264.path,
            device_config,
            num_frames=5,
        )

        batches = list(streamer)
        total_frames = sum(
            batch.__cuda_array_interface__["shape"][0] for batch in batches
        )

        # 100 frames with batch size of 5 should give 20 batches
        self.assertEqual(total_frames, 100)
        self.assertEqual(len(batches), 20)

        # Each batch should have 5 frames
        for batch in batches:
            self.assertEqual(batch.__cuda_array_interface__["shape"][0], 5)

    def test_streaming_load_video_nvdec_to_rgb(self) -> None:
        """Can convert NV12 frames to RGB during streaming"""
        h264 = _get_h264_sample()

        device_config = spdl.io.cuda_config(device_index=DEFAULT_CUDA)
        streamer = spdl.io.streaming_load_video_nvdec(
            h264.path,
            device_config,
            num_frames=32,
        )

        # Get first batch and convert to RGB
        first_batch = next(iter(streamer))
        num_frames = first_batch.__cuda_array_interface__["shape"][0]
        self.assertEqual(num_frames, 32)  # 32 frames in batch

        # Convert batched NV12 to RGB using the batched conversion function
        rgb_buffer = spdl.io.nv12_to_rgb(first_batch, device_config=device_config)
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

        batches = list(streamer)

        # Should get 1 batch with all 100 frames
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].__cuda_array_interface__["shape"][0], 100)

    def test_streaming_load_video_nvdec_decoder_cache(self) -> None:
        """Using cached decoder can decode frames properly"""

        # Setup: Create test video sample
        h264 = _get_h264_sample()
        cuda_config = spdl.io.cuda_config(device_index=DEFAULT_CUDA)

        packets = spdl.io.demux_video(h264.path)
        packets = spdl.io.apply_bsf(packets, "h264_mp4toannexb")
        codec = packets.codec
        assert codec is not None
        decoder = spdl.io.nvdec_decoder(cuda_config, codec, use_cache=False)
        ref = spdl.io.to_torch(decoder.decode_packets(packets.clone()))

        packets = spdl.io.demux_video(h264.path, timestamp=(1.0, 2.0))
        packets = spdl.io.apply_bsf(packets, "h264_mp4toannexb")
        codec = packets.codec
        assert codec is not None
        decoder = spdl.io.nvdec_decoder(cuda_config, codec, use_cache=False)
        _ = spdl.io.to_torch(decoder.decode_packets(packets.clone()))
        # implementation detail: at this point, the timestamp is stored in decoder.
        # it must be internally reset before the next decoding.

        tensors = []
        for buffer in spdl.io.streaming_load_video_nvdec(
            h264.path, cuda_config, num_frames=16
        ):
            tensors.append(spdl.io.to_torch(buffer))
        hyp = torch.cat(tensors)

        print(f"{hyp.shape=}")
        print(f"{ref.shape=}")
        torch.testing.assert_close(hyp, ref)
