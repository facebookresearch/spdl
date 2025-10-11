# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import struct
import unittest

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

    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    data_size = num_samples * block_align
    byte_rate = sample_rate * block_align

    fmt_chunk = struct.pack(
        "<4sI2H2I2H",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )

    data_chunk_header = struct.pack("<4sI", b"data", data_size)

    # Create reference waveform as numpy array
    dtype_map = {
        8: np.uint8,
        16: np.int16,
        32: np.int32,
        64: np.float64,
    }
    dtype = dtype_map[bits_per_sample]

    # Generate sample data
    # Use a pattern that's easy to verify: incrementing values mod 256 for raw bytes
    reference = np.zeros((num_samples, num_channels), dtype=dtype)
    for sample_idx in range(num_samples):
        for channel_idx in range(num_channels):
            # Create a pattern based on sample and channel index
            value = (sample_idx * num_channels + channel_idx) % 256

            # Convert to appropriate dtype value
            if bits_per_sample == 8:
                reference[sample_idx, channel_idx] = value
            elif bits_per_sample == 16:
                # For 16-bit, use lower byte for LSB, value for MSB to create pattern
                reference[sample_idx, channel_idx] = value | (value << 8)
            elif bits_per_sample == 32:
                reference[sample_idx, channel_idx] = (
                    value | (value << 8) | (value << 16) | (value << 24)
                )
            elif bits_per_sample == 64:
                reference[sample_idx, channel_idx] = float(value)

    # Convert reference to bytes
    samples = reference.tobytes()

    riff_size = 4 + len(fmt_chunk) + len(data_chunk_header) + data_size
    riff_header = struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE")

    wav_data = riff_header + fmt_chunk + data_chunk_header + samples

    return wav_data, reference


class WAVUtilsTest(unittest.TestCase):
    def test_parse_wav_header(self):
        wav_data, _ = create_wav_data(
            sample_rate=8000, num_channels=2, bits_per_sample=16, num_samples=8000
        )

        header = spdl.io.parse_wav_header(wav_data)

        self.assertEqual(header.audio_format, 1)
        self.assertEqual(header.num_channels, 2)
        self.assertEqual(header.sample_rate, 8000)
        self.assertEqual(header.bits_per_sample, 16)
        self.assertEqual(header.block_align, 4)
        self.assertEqual(header.byte_rate, 32000)
        self.assertEqual(header.data_size, 32000)

    def test_extract_full_waveform(self):
        wav_data, reference = create_wav_data(num_samples=100)

        samples = spdl.io.load_wav(wav_data)

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
        samples = spdl.io.load_wav(wav_data, time_offset_seconds=time_offset)

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
        samples = spdl.io.load_wav(
            wav_data, time_offset_seconds=time_offset, duration_seconds=duration
        )

        # Calculate expected slice
        start_sample = int(time_offset * sample_rate)
        end_sample = start_sample + int(duration * sample_rate)
        expected_reference = reference[start_sample:end_sample]

        # Validate shape and dtype
        self.assertEqual(samples.shape, expected_reference.shape)
        self.assertEqual(samples.dtype, np.int16)

        # Validate against reference waveform
        np.testing.assert_array_equal(samples, expected_reference)

    def test_invalid_wav_data(self):
        invalid_data = b"not a wav file"

        with self.assertRaises(ValueError):
            spdl.io.parse_wav_header(invalid_data)

    def test_wav_data_too_small(self):
        too_small = b"RIFF\x00\x00\x00\x00WAVE"

        with self.assertRaises(ValueError):
            spdl.io.parse_wav_header(too_small)

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

    def test_repr(self):
        wav_data, _ = create_wav_data()
        header = spdl.io.parse_wav_header(wav_data)

        repr_str = repr(header)

        self.assertIn("WAVHeader", repr_str)
        self.assertIn("sample_rate=8000", repr_str)
        self.assertIn("num_channels=2", repr_str)

    def test_load_wav_returns_numpy_array(self):
        wav_data, reference = create_wav_data(
            sample_rate=8000, num_channels=2, bits_per_sample=16, num_samples=100
        )

        samples = spdl.io.load_wav(wav_data)

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

        # Test header parsing
        header = spdl.io.parse_wav_header(wav_data)
        self.assertEqual(header.num_channels, num_channels)
        self.assertEqual(header.sample_rate, sample_rate)
        self.assertEqual(header.bits_per_sample, 16)

        # Test sample extraction
        samples = spdl.io.load_wav(wav_data)
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
        samples = spdl.io.load_wav(
            wav_data, time_offset_seconds=time_offset, duration_seconds=duration
        )

        # Calculate expected slice
        start_sample = int(time_offset * sample_rate)
        end_sample = start_sample + int(duration * sample_rate)
        expected_reference = reference[start_sample:end_sample]

        self.assertEqual(samples.shape, expected_reference.shape)
        self.assertEqual(samples.dtype, np.int16)

        # Validate against reference waveform
        np.testing.assert_array_equal(samples, expected_reference)
