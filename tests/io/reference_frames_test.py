# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import gc
import unittest
import weakref

import numpy as np
import spdl.io


class TestReferenceAudioFrame(unittest.TestCase):
    def test_audio_frame_keeps_reference_alive(self) -> None:
        """Test that AudioFrame keeps the original array alive even after deletion."""
        # Setup: Create an array and a weak reference to it
        array = np.random.randint(0, 255, size=(100, 2), dtype=np.uint8)
        weak_ref = weakref.ref(array)

        # Execute: Create a reference frame
        frame = spdl.io.create_reference_audio_frame(
            array=array,
            sample_fmt="u8",
            sample_rate=44100,
            pts=0,
        )

        # Assert: The array should still be alive (referenced by both array and frame)
        self.assertIsNotNone(weak_ref())

        # Execute: Delete the original array variable
        del array
        gc.collect()

        # Assert: The array should still be alive (referenced by frame)
        self.assertIsNotNone(weak_ref())

        # Execute: Delete the frame
        del frame
        gc.collect()

        # Assert: Now the array should be garbage collected
        self.assertIsNone(weak_ref())

    def test_audio_frame_can_be_converted(self) -> None:
        """Test that reference audio frame can be converted to buffer."""
        array = np.random.randint(0, 255, size=(100, 2), dtype=np.uint8)

        frame = spdl.io.create_reference_audio_frame(
            array=array,
            sample_fmt="u8",
            sample_rate=44100,
            pts=0,
        )
        buffer = spdl.io.convert_frames(frame)

        buffer_array = spdl.io.to_numpy(buffer)
        self.assertEqual(buffer_array.shape, (100, 2))
        np.testing.assert_array_equal(buffer_array, array)

    def test_audio_frame_planar_format(self) -> None:
        """Test that planar format audio frames work correctly."""
        # Setup: Create planar format data (channels first)
        array = np.random.randint(0, 255, size=(2, 100), dtype=np.uint8)

        # Execute: Create reference frame with planar format
        frame = spdl.io.create_reference_audio_frame(
            array=array,
            sample_fmt="u8p",
            sample_rate=44100,
            pts=0,
        )

        # Assert: Frame should have correct properties
        self.assertEqual(frame.num_frames, 100)
        self.assertEqual(frame.num_channels, 2)
        self.assertEqual(frame.sample_rate, 44100)


class TestReferenceVideoFrame(unittest.TestCase):
    def test_video_frame_keeps_reference_alive(self) -> None:
        """Test that VideoFrame keeps the original array alive even after deletion."""
        # Setup: Create an array and a weak reference to it
        array = np.random.randint(0, 255, size=(5, 128, 128, 3), dtype=np.uint8)
        weak_ref = weakref.ref(array)

        # Execute: Create a reference frame
        frame = spdl.io.create_reference_video_frame(
            array=array,
            pix_fmt="rgb24",
            frame_rate=(30, 1),
            pts=0,
        )

        # Assert: The array should still be alive (referenced by both array and frame)
        self.assertIsNotNone(weak_ref())

        # Execute: Delete the original array variable
        del array
        gc.collect()

        # Assert: The array should still be alive (referenced by frame)
        self.assertIsNotNone(weak_ref())

        # Execute: Delete the frame
        del frame
        gc.collect()

        # Assert: Now the array should be garbage collected
        self.assertIsNone(weak_ref())

    def test_video_frame_can_be_converted(self) -> None:
        """Test that reference video frame can be converted to buffer."""
        array = np.random.randint(0, 255, size=(5, 128, 128, 3), dtype=np.uint8)

        frame = spdl.io.create_reference_video_frame(
            array=array,
            pix_fmt="rgb24",
            frame_rate=(30, 1),
            pts=0,
        )
        buffer = spdl.io.convert_frames(frame)

        buffer_array = spdl.io.to_numpy(buffer)
        self.assertEqual(buffer_array.shape, (5, 128, 128, 3))
        np.testing.assert_array_equal(buffer_array, array)

    def test_video_frame_grayscale(self) -> None:
        """Test that grayscale video frames work correctly."""
        # Setup: Create grayscale data
        array = np.random.randint(0, 255, size=(5, 128, 128), dtype=np.uint8)

        # Execute: Create reference frame with grayscale format
        frame = spdl.io.create_reference_video_frame(
            array=array,
            pix_fmt="gray8",
            frame_rate=(30, 1),
            pts=0,
        )

        # Assert: Frame should have correct properties
        self.assertEqual(frame.num_frames, 5)
        self.assertEqual(frame.width, 128)
        self.assertEqual(frame.height, 128)

    def test_video_frame_yuv_format(self) -> None:
        """Test that YUV planar format video frames work correctly."""
        # Setup: Create YUV planar format data (channels first)
        array = np.random.randint(0, 255, size=(5, 3, 128, 128), dtype=np.uint8)

        # Execute: Create reference frame with YUV planar format
        frame = spdl.io.create_reference_video_frame(
            array=array,
            pix_fmt="yuv444p",
            frame_rate=(30, 1),
            pts=0,
        )

        # Assert: Frame should have correct properties
        self.assertEqual(frame.num_frames, 5)
        self.assertEqual(frame.width, 128)
        self.assertEqual(frame.height, 128)
