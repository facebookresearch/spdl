import numpy as np

from spdl._internal import import_utils

from spdl.io import CPUBuffer, CUDABuffer

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


def to_numpy(buffer: CPUBuffer) -> NDArray:
    """Convert to NumPy NDArray.

    Args:
        buffer: Object implements the array interface protocol.

    Returns:
        (NDArray): A NumPy array.

    See also:
        https://numpy.org/doc/stable/reference/arrays.interface.html
    """
    if not hasattr(buffer, "__array_interface__"):
        raise RuntimeError(
            "The given object does not have `__array_interface__` attribute."
        )
    return np.array(buffer, copy=False)


def to_torch(buffer: CPUBuffer | CUDABuffer):
    """Convert to PyTorch Tensor.

    Args:
        buffer: Object implements the (CUDA) array interface protocol.

    Returns:
        (torch.Tensor): A PyTorch Tensor.
    """
    if hasattr(buffer, "__cuda_array_interface__"):
        data_ptr = buffer.__cuda_array_interface__["data"][0]
        tensor = torch.as_tensor(buffer, device=f"cuda:{buffer.device_index}")
        if tensor.data_ptr() == 0:
            raise RuntimeError(
                "Failed to convert to PyTorch Tensor. "
                f"src: {data_ptr}, dst: {tensor.data_ptr()}, device: {buffer.device_index}"
            )
        if tensor.data_ptr() != data_ptr:
            raise RuntimeError(
                "[INTERNAL ERROR] Failed to perform zero-copy conversion to PyTorch Tensor. "
                f"src: {data_ptr}, dst: {tensor.data_ptr()}, device: {buffer.device_index}"
            )
        return tensor

    # Not sure how to make as_tensor work with __array_interface__.
    # Using numpy as intermediate.
    return torch.as_tensor(np.array(buffer, copy=False))


def to_numba(buffer: CPUBuffer | CUDABuffer):
    """Convert to Numba DeviceNDArray or NumPy NDArray.

    Args:
        buffer: Object implements the (CUDA) array interface protocol.

    Returns:
        (DeviceNDArray or NDArray): A Numba DeviceNDArray or NumPy NDArray.

    See Also:
        https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
    """
    if hasattr(buffer, "__cuda_array_interface__"):
        return cuda.as_cuda_array(buffer)

    return np.array(buffer, copy=False)
