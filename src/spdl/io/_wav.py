# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""WAV audio processing utilities."""

# Importing `spdl.io.lib` instead of `spdl.io.lilb._archive`
# so as to delay the import of C++ extension module
from . import lib as _libspdl

__all__ = [
    "load_wav",
]


def __dir__():
    return __all__


class WAVArrayInterface:
    """Wrapper class that exposes WAV data through the Array Interface Protocol.

    This class holds the array interface dictionary returned by the C++ binding
    and exposes it through the __array_interface__ property, making it compatible
    with NumPy and other libraries that support the array interface protocol.
    """

    def __init__(self, array_interface_dict: dict) -> None:
        """Initialize with an array interface dictionary.

        Args:
            array_interface_dict: Dictionary containing array interface metadata
                from the C++ binding, including version, shape, typestr, data, and owner.
        """
        self._array_interface = array_interface_dict

    @property
    def __array_interface__(self) -> dict:
        """Return the array interface dictionary for NumPy compatibility.

        Returns:
            Dictionary containing array interface metadata:
                - version: Protocol version (3)
                - shape: Tuple of array dimensions
                - typestr: Data type string
                - data: Tuple of (data_pointer, read_only_flag)
                - owner: Object owning the data buffer
        """
        return self._array_interface


def load_wav(
    data: bytes,
    time_offset_seconds: float | None = None,
    duration_seconds: float | None = None,
) -> WAVArrayInterface:
    """Extract audio samples from WAV data.

    Args:
        data: Binary WAV data as bytes
        time_offset_seconds: Optional starting time in seconds (default: 0.0)
        duration_seconds: Optional duration in seconds (default: until end)

    Returns:
        WAVArrayInterface: Object exposing audio samples through the Array Interface Protocol.
            The object can be consumed by NumPy (np.asarray) and other libraries supporting
            the array interface. The underlying array has shape (num_samples, num_channels).
            The dtype depends on bits_per_sample:

               - 8 bits: uint8
               - 16 bits: int16
               - 32 bits: int32 or float32
               - 64 bits: float64

    Raises:
        ValueError: If the WAV data is invalid or time range is out of bounds.

    .. seealso::

       :ref:example-benchmark-wav

          A benchmark script that compares the performance of ``load_wav`` function with
          :py:func:`load_audio` and ``libsoundfile``.
    """
    array_interface_dict = _libspdl._wav.load_wav(
        data,
        time_offset_seconds=time_offset_seconds,
        duration_seconds=duration_seconds,
    )
    return WAVArrayInterface(array_interface_dict)
