from typing import Optional

import numpy as np

from spdl import libspdl
from spdl._internal import import_utils

try:
    from numpy.typing import NDArray
except ImportError:
    from numpy import ndarray as NDArray

torch = import_utils.lazy_import("torch")
cuda = import_utils.lazy_import("numba.cuda")

__all__ = [
    "to_numba",
    "to_numpy",
    "to_torch",
]


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
    """Convert to NumPy NDArray.

    Args:
        frames (DecodedFrames): Decoded frames.

        index (int or None): The index of plane to be included in the output.
            For formats like YUV420, in which chroma planes have different sizes
            than luma plane, this argument can be used to select a specific plane.
    """
    return np.array(_to_cpu_buffer(frames, index), copy=False)


def to_torch(frames, index: Optional[int] = None):
    """Convert to PyTorch Tensor.

    Args:
        frames (DecodedFrames): Decoded frames.

        index (int or None): The index of plane to be included in the output.
            For formats like YUV420, in which chroma planes have different sizes
            than luma plane, this argument can be used to select a specific plane.
    """
    buffer = _BufferWrapper(libspdl.convert_frames(frames, index))

    if buffer.is_cuda:
        data_ptr = buffer.__cuda_array_interface__["data"][0]
        index = libspdl.get_cuda_device_index(data_ptr)
        tensor = torch.as_tensor(buffer, device=f"cuda:{index}")
        assert (
            tensor.data_ptr() == data_ptr
        ), "[INTERNAL ERROR] Failed to perform zero-copy conversion to PyTorch Tensor."
        return tensor

    # Not sure how to make as_tensor work with __array_interface__.
    # Using numpy as intermediate.
    return torch.as_tensor(np.array(buffer, copy=False))


def to_numba(frames, index: Optional[int] = None):
    """Convert to Numba DeviceNDArray or NumPy NDArray.

    Args:
        frames (DecodedFrames): Decoded frames.

        index (int or None): The index of plane to be included in the output.
            For formats like YUV420, in which chroma planes have different sizes
            than luma plane, this argument can be used to select a specific plane.
    """
    buffer = _BufferWrapper(libspdl.convert_frames(frames, index))

    if buffer.is_cuda:
        return cuda.as_cuda_array(buffer)

    return np.array(buffer, copy=False)
