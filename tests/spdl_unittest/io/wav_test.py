# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import io
import unittest
import wave

import numpy as np
import spdl.io
from parameterized import parameterized


def create_wav_data(
    sample_rate: int = 8000,
    num_channels: int = 2,
    bits_per_sample: int = 16,
    num_samples: int = 8000,
) -> "tuple[bytes, np.ndarray]":
    """Create a simple WAV file in memory for testing.

    Returns:
        tuple: (wav_data, reference_waveform)
            - wav_data: Complete WAV file as bytes
            - reference_waveform: numpy array with shape (num_samples, num_channels)
                                  containing the expected audio samples
    """

    dtype_map = {
        8: np.uint8,
        16: np.int16,
        32: np.int32,
        64: np.float64,
    }
    dtype = dtype_map[bits_per_sample]

    reference = np.zeros((num_samples, num_channels), dtype=dtype)
    if bits_per_sample == 64:
        for channel_idx in range(num_channels):
            reference[:, channel_idx] = np.linspace(-1.0, 1.0, num_samples, dtype=dtype)
    else:
        # For integer types, use linspace to cover full dtype range
        dtype_info = np.iinfo(dtype)
        for channel_idx in range(num_channels):
            reference[:, channel_idx] = np.linspace(
                dtype_info.min, dtype_info.max, num_samples, dtype=dtype
            )

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(bits_per_sample // 8)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(reference.tobytes())

    wav_data = wav_buffer.getvalue()
    return wav_data, reference


class WAVUtilsTest(unittest.TestCase):
    def test_extract_full_waveform(self):
        wav_data, reference = create_wav_data(num_samples=100)

        wav_array_interface = spdl.io.load_wav(wav_data)
        samples = spdl.io.to_numpy(wav_array_interface)

        # Validate shape and dtype
        self.assertEqual(samples.shape, (100, 2))
        self.assertEqual(samples.dtype, np.int16)

        # Validate against reference waveform
        np.testing.assert_array_equal(samples, reference)

    def test_extract_with_time_offset(self):
        sample_rate = 8000
        num_channels = 2
        bits_per_sample = 16
        duration_seconds = 2.0
        num_samples = int(sample_rate * duration_seconds)

        wav_data, reference = create_wav_data(
            sample_rate=sample_rate,
            num_channels=num_channels,
            bits_per_sample=bits_per_sample,
            num_samples=num_samples,
        )

        time_offset = 0.5
        wav_array_interface = spdl.io.load_wav(
            wav_data, time_offset_seconds=time_offset
        )
        samples = spdl.io.to_numpy(wav_array_interface)

        # Calculate expected slice
        start_sample = int(time_offset * sample_rate)
        expected_reference = reference[start_sample:]

        # Validate shape and dtype
        self.assertEqual(samples.shape, expected_reference.shape)
        self.assertEqual(samples.dtype, np.int16)

        # Validate against reference waveform
        np.testing.assert_array_equal(samples, expected_reference)

    def test_extract_with_offset_and_duration(self):
        sample_rate = 8000
        num_channels = 2
        bits_per_sample = 16
        total_duration = 3.0
        num_samples = int(sample_rate * total_duration)

        wav_data, reference = create_wav_data(
            sample_rate=sample_rate,
            num_channels=num_channels,
            bits_per_sample=bits_per_sample,
            num_samples=num_samples,
        )

        time_offset = 1.0
        duration = 1.0
        wav_array_interface = spdl.io.load_wav(
            wav_data, time_offset_seconds=time_offset, duration_seconds=duration
        )
        samples = spdl.io.to_numpy(wav_array_interface)

        # Calculate expected slice
        start_sample = int(time_offset * sample_rate)
        end_sample = start_sample + int(duration * sample_rate)
        expected_reference = reference[start_sample:end_sample]

        # Validate shape and dtype
        self.assertEqual(samples.shape, expected_reference.shape)
        self.assertEqual(samples.dtype, np.int16)

        # Validate against reference waveform
        np.testing.assert_array_equal(samples, expected_reference)

    def test_negative_time_offset(self):
        wav_data, _ = create_wav_data(num_samples=8000)

        with self.assertRaises(ValueError):
            spdl.io.load_wav(wav_data, time_offset_seconds=-1.0)

    def test_negative_duration(self):
        wav_data, _ = create_wav_data(num_samples=8000)

        with self.assertRaises(ValueError):
            spdl.io.load_wav(wav_data, duration_seconds=-1.0)

    def test_time_offset_exceeds_duration(self):
        wav_data, _ = create_wav_data(sample_rate=8000, num_samples=8000)

        with self.assertRaises(ValueError):
            spdl.io.load_wav(wav_data, time_offset_seconds=10.0)

    def test_load_wav_returns_numpy_array(self):
        wav_data, reference = create_wav_data(
            sample_rate=8000, num_channels=2, bits_per_sample=16, num_samples=100
        )

        wav_array_interface = spdl.io.load_wav(wav_data)
        samples = spdl.io.to_numpy(wav_array_interface)

        self.assertEqual(samples.shape, (100, 2))
        self.assertEqual(samples.dtype, np.int16)
        self.assertFalse(samples.flags["OWNDATA"])

        # Validate against reference waveform
        np.testing.assert_array_equal(samples, reference)

    @parameterized.expand(
        [
            (1,),
            (2,),
            (3,),
            (4,),
            (8,),
            (16,),
            (32,),
        ]
    )
    def test_load_wav_multi_channel(self, num_channels):
        sample_rate = 8000
        num_samples = 1000

        wav_data, reference = create_wav_data(
            sample_rate=sample_rate,
            num_channels=num_channels,
            bits_per_sample=16,
            num_samples=num_samples,
        )

        # Test sample extraction
        wav_array_interface = spdl.io.load_wav(wav_data)
        samples = spdl.io.to_numpy(wav_array_interface)
        self.assertEqual(samples.shape, (num_samples, num_channels))
        self.assertEqual(samples.dtype, np.int16)
        self.assertFalse(samples.flags["OWNDATA"])

        # Validate against reference waveform
        np.testing.assert_array_equal(samples, reference)

    @parameterized.expand(
        [
            (3,),
            (4,),
            (8,),
            (16,),
            (32,),
        ]
    )
    def test_load_wav_multi_channel_with_time_window(self, num_channels):
        sample_rate = 8000
        total_duration = 2.0
        num_samples = int(sample_rate * total_duration)

        wav_data, reference = create_wav_data(
            sample_rate=sample_rate,
            num_channels=num_channels,
            bits_per_sample=16,
            num_samples=num_samples,
        )

        # Extract 1 second starting at 0.5 seconds
        time_offset = 0.5
        duration = 1.0
        wav_array_interface = spdl.io.load_wav(
            wav_data, time_offset_seconds=time_offset, duration_seconds=duration
        )
        samples = spdl.io.to_numpy(wav_array_interface)

        # Calculate expected slice
        start_sample = int(time_offset * sample_rate)
        end_sample = start_sample + int(duration * sample_rate)
        expected_reference = reference[start_sample:end_sample]

        self.assertEqual(samples.shape, expected_reference.shape)
        self.assertEqual(samples.dtype, np.int16)

        # Validate against reference waveform
        np.testing.assert_array_equal(samples, expected_reference)
