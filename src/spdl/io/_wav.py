# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""WAV audio processing utilities."""

from dataclasses import dataclass

# Importing `spdl.io.lib` instead of `spdl.io.lilb._archive`
# so as to delay the import of C++ extension module
from . import lib as _libspdl

__all__ = [
    "parse_wav_header",
    "load_wav",
]


def __dir__():
    return __all__


@dataclass
class _WAVHeader:
    """WAV file header information."""

    audio_format: int
    """Audio format code (1=PCM, 3=IEEE float, etc.)"""

    num_channels: int
    """Number of audio channels."""

    sample_rate: int
    """Sample rate in Hz."""

    byte_rate: int
    """Average bytes per second."""

    block_align: int
    """Block alignment in bytes."""

    bits_per_sample: int
    """Bits per sample"""

    data_size: int
    """data_size"""


def parse_wav_header(data: bytes) -> _WAVHeader:
    """Parse WAV header from audio data.

    Args:
        data: Binary WAV data as bytes

    Returns:
        A data class represents the parsed header information.
        The following attributes are defined.

          - ``audio_format``: Audio format code (1=PCM, 3=IEEE float, etc.)
          - ``num_channels``: Number of audio channels
          - ``sample_rate``: Sample rate in Hz
          - ``byte_rate``: Average bytes per second
          - ``block_align``: Block alignment in bytes
          - ``bits_per_sample``: Bits per sample
          - ``data_size``: Size of audio data in bytes

    Raises:
        ValueError: If the WAV data is invalid or cannot be parsed
    """
    result = _libspdl._wav.parse_wav_header(data)
    return _WAVHeader(*result)


def load_wav(
    data: bytes,
    time_offset_seconds: float | None = None,
    duration_seconds: float | None = None,
):
    """Extract audio samples from WAV data as numpy array.

    Args:
        wav_data: Binary WAV data as bytes or string
        time_offset_seconds: Optional starting time in seconds (default: 0.0)
        duration_seconds: Optional duration in seconds (default: until end)

    Returns:
        ndarray: Audio samples as numpy array with shape (num_samples, num_channels)
            The dtype depends on bits_per_sample:

               - 8 bits: uint8
               - 16 bits: int16
               - 32 bits: int32 or float32
               - 64 bits: float64
    Raises:
        ValueError: If the WAV data is invalid or time range is out of bounds.
    """
    return _libspdl._wav.load_wav(
        data,
        time_offset_seconds=time_offset_seconds,
        duration_seconds=duration_seconds,
    )
