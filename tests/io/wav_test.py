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


class ParseWAVTest(unittest.TestCase):
    def test_parse_wav_basic_metadata(self):
        """Test that parse_wav extracts correct basic metadata from a WAV file."""
        # Setup: Create a simple WAV file with known parameters
        sample_rate = 8000
        num_channels = 2
        bits_per_sample = 16
        num_samples = 1000

        wav_data, _ = create_wav_data(
            sample_rate=sample_rate,
            num_channels=num_channels,
            bits_per_sample=bits_per_sample,
            num_samples=num_samples,
        )

        # Execute: Parse the WAV header
        header = spdl.io.parse_wav(wav_data)

        # Assert: Verify all header fields are correct
        self.assertEqual(header.sample_rate, sample_rate)
        self.assertEqual(header.num_channels, num_channels)
        self.assertEqual(header.bits_per_sample, bits_per_sample)

        # Verify calculated fields
        expected_byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
        expected_block_align = num_channels * (bits_per_sample // 8)
        expected_data_size = num_samples * expected_block_align

        self.assertEqual(header.byte_rate, expected_byte_rate)
        self.assertEqual(header.block_align, expected_block_align)
        self.assertEqual(header.data_size, expected_data_size)

    def test_parse_wav_audio_format_pcm(self):
        """Test that parse_wav correctly identifies PCM audio format."""
        # Setup: Create a PCM WAV file (16-bit PCM has audio_format=1)
        wav_data, _ = create_wav_data(bits_per_sample=16)

        # Execute: Parse the WAV header
        header = spdl.io.parse_wav(wav_data)

        # Assert: Verify audio_format is 1 (PCM)
        self.assertEqual(header.audio_format, 1)

    @parameterized.expand(
        [
            (1,),
            (2,),
            (4,),
            (8,),
        ]
    )
    def test_parse_wav_multi_channel(self, num_channels):
        """Test that parse_wav correctly handles different channel counts."""
        # Setup: Create a WAV file with specific number of channels
        wav_data, _ = create_wav_data(num_channels=num_channels, num_samples=100)

        # Execute: Parse the WAV header
        header = spdl.io.parse_wav(wav_data)

        # Assert: Verify num_channels is correct
        self.assertEqual(header.num_channels, num_channels)

    @parameterized.expand(
        [
            (8000,),
            (16000,),
            (22050,),
            (44100,),
            (48000,),
        ]
    )
    def test_parse_wav_sample_rates(self, sample_rate):
        """Test that parse_wav correctly handles different sample rates."""
        # Setup: Create a WAV file with specific sample rate
        wav_data, _ = create_wav_data(sample_rate=sample_rate, num_samples=100)

        # Execute: Parse the WAV header
        header = spdl.io.parse_wav(wav_data)

        # Assert: Verify sample_rate is correct
        self.assertEqual(header.sample_rate, sample_rate)

    @parameterized.expand(
        [
            (8,),
            (16,),
            (32,),
        ]
    )
    def test_parse_wav_bit_depths(self, bits_per_sample):
        """Test that parse_wav correctly handles different bit depths."""
        # Setup: Create a WAV file with specific bit depth
        wav_data, _ = create_wav_data(bits_per_sample=bits_per_sample, num_samples=100)

        # Execute: Parse the WAV header
        header = spdl.io.parse_wav(wav_data)

        # Assert: Verify bits_per_sample is correct
        self.assertEqual(header.bits_per_sample, bits_per_sample)

    def test_parse_wav_returns_wav_header(self):
        """Test that parse_wav returns a WAVHeader with all required attributes."""
        # Setup: Create a simple WAV file
        wav_data, _ = create_wav_data(num_samples=100)

        # Execute: Parse the WAV header
        header = spdl.io.parse_wav(wav_data)

        # Assert: Verify return type is WAVHeader and contains all expected attributes
        self.assertIsInstance(header, spdl.io.WAVHeader)

        # Verify all expected attributes exist and are accessible
        self.assertTrue(hasattr(header, "audio_format"))
        self.assertTrue(hasattr(header, "num_channels"))
        self.assertTrue(hasattr(header, "sample_rate"))
        self.assertTrue(hasattr(header, "byte_rate"))
        self.assertTrue(hasattr(header, "block_align"))
        self.assertTrue(hasattr(header, "bits_per_sample"))
        self.assertTrue(hasattr(header, "data_size"))

    def test_parse_wav_invalid_data(self):
        """Test that parse_wav raises ValueError for invalid WAV data."""
        # Setup: Create invalid WAV data (not a real WAV file)
        invalid_data = b"This is not a WAV file"

        # Execute & Assert: Verify that parsing invalid data raises ValueError
        with self.assertRaises(ValueError):
            spdl.io.parse_wav(invalid_data)

    def test_parse_wav_empty_data(self):
        """Test that parse_wav raises ValueError for empty data."""
        # Setup: Create empty bytes
        empty_data = b""

        # Execute & Assert: Verify that parsing empty data raises ValueError
        with self.assertRaises(ValueError):
            spdl.io.parse_wav(empty_data)

    def test_parse_wav_consistency_with_load_wav(self):
        """Test that parse_wav metadata matches the actual loaded audio data."""
        # Setup: Create a WAV file with known parameters
        sample_rate = 16000
        num_channels = 2
        bits_per_sample = 16
        num_samples = 500

        wav_data, _ = create_wav_data(
            sample_rate=sample_rate,
            num_channels=num_channels,
            bits_per_sample=bits_per_sample,
            num_samples=num_samples,
        )

        # Execute: Parse header and load audio data
        header = spdl.io.parse_wav(wav_data)
        wav_array_interface = spdl.io.load_wav(wav_data)
        samples = spdl.io.to_numpy(wav_array_interface)

        # Assert: Verify header metadata matches actual loaded data
        self.assertEqual(header.num_channels, samples.shape[1])
        self.assertEqual(header.sample_rate, sample_rate)

        # Calculate expected samples from header metadata
        expected_num_samples = header.data_size // header.block_align
        self.assertEqual(expected_num_samples, samples.shape[0])

    def test_parse_wav_byte_rate_calculation(self):
        """Test that parse_wav correctly calculates byte_rate."""
        # Setup: Create a WAV file with specific parameters
        sample_rate = 44100
        num_channels = 2
        bits_per_sample = 16

        wav_data, _ = create_wav_data(
            sample_rate=sample_rate,
            num_channels=num_channels,
            bits_per_sample=bits_per_sample,
            num_samples=100,
        )

        # Execute: Parse the WAV header
        header = spdl.io.parse_wav(wav_data)

        # Assert: Verify byte_rate calculation
        # byte_rate = sample_rate * num_channels * (bits_per_sample / 8)
        expected_byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
        self.assertEqual(header.byte_rate, expected_byte_rate)

    def test_parse_wav_block_align_calculation(self):
        """Test that parse_wav correctly calculates block_align."""
        # Setup: Create a WAV file with specific parameters
        num_channels = 4
        bits_per_sample = 16

        wav_data, _ = create_wav_data(
            num_channels=num_channels,
            bits_per_sample=bits_per_sample,
            num_samples=100,
        )

        # Execute: Parse the WAV header
        header = spdl.io.parse_wav(wav_data)

        # Assert: Verify block_align calculation
        # block_align = num_channels * (bits_per_sample / 8)
        expected_block_align = num_channels * (bits_per_sample // 8)
        self.assertEqual(header.block_align, expected_block_align)

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
