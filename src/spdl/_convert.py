from typing import Optional

import numpy as np

from spdl import libspdl

try:
    from numpy.typing import NDArray
except ImportError:
    from numpy import ndarray as NDArray


class _BufferWrapper:
    def __init__(self, buffer):
        self._buffer = buffer

    def __getattr__(self, name):
        if name == "__array_interface__":
            if not self._buffer.is_cuda:
                return self._buffer.get_array_interface()
        if name == "__cuda_array_interface__":
            if self._buffer.is_cuda:
                return self._buffer.get_cuda_array_interface()
        return getattr(self._buffer, name)


def _to_cpu_buffer(frames, index=None):
    return _BufferWrapper(libspdl.convert_to_cpu_buffer(frames, index))


def to_numpy(frames, index: Optional[int] = None) -> NDArray:
    """Convert to numpy array.

    Args:
        frames (DecodedFrames): Decoded frames.

        index (int or None): The index of plane to be included in the output.
            For formats like YUV420, in which chroma planes have different sizes
            than luma plane, this argument can be used to select a specific plane.
    """
    return np.array(_to_cpu_buffer(frames, index), copy=False)
