import numpy as np

from spdl._internal import import_utils
from spdl.lib import _libspdl

from ._type_stub import Buffer

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


def to_numpy(buffer: Buffer) -> NDArray:
    """Convert to NumPy NDArray.

    Args:
        buffer: Object implements the array interface protocol.

    Returns:
        (NDArray): A NumPy array.

    See also:
        https://numpy.org/doc/stable/reference/arrays.interface.html
    """
    if buffer.is_cuda:
        raise RuntimeError("to_numpy() does not support CUDA buffers.")
    return np.array(buffer, copy=False)


def to_torch(buffer: Buffer):
    """Convert to PyTorch Tensor.

    Args:
        buffer: Object implements the (CUDA) array interface protocol.

    Returns:
        (torch.Tensor): A PyTorch Tensor.
    """
    if buffer.is_cuda:
        data_ptr = buffer.__cuda_array_interface__["data"][0]
        index = buffer.device_index
        tensor = torch.as_tensor(buffer, device=f"cuda:{index}")
        if tensor.data_ptr() == 0:
            raise RuntimeError(
                "Failed to convert to PyTorch Tensor. "
                f"src: {data_ptr}, dst: {tensor.data_ptr()}, device: {index}"
            )
        if tensor.data_ptr() != data_ptr:
            raise RuntimeError(
                "[INTERNAL ERROR] Failed to perform zero-copy conversion to PyTorch Tensor. "
                f"src: {data_ptr}, dst: {tensor.data_ptr()}, device: {index}"
            )
        return tensor

    # Not sure how to make as_tensor work with __array_interface__.
    # Using numpy as intermediate.
    return torch.as_tensor(np.array(buffer, copy=False))


def to_numba(buffer: Buffer):
    """Convert to Numba DeviceNDArray or NumPy NDArray.

    Args:
        buffer: Object implements the (CUDA) array interface protocol.

    Returns:
        (DeviceNDArray or NDArray): A Numba DeviceNDArray or NumPy NDArray.

    See Also:
        https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
    """
    if buffer.is_cuda:
        return cuda.as_cuda_array(buffer)

    return np.array(buffer, copy=False)
