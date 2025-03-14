# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from spdl.io import CPUBuffer, CUDABuffer

from ._internal import import_utils

torch = import_utils.lazy_import("torch")
cuda = import_utils.lazy_import("numba.cuda")
jax = import_utils.lazy_import("jax")
np = import_utils.lazy_import("numpy")

__all__ = [
    "to_numba",
    "to_numpy",
    "to_torch",
    "to_jax",
]


def to_numpy(buffer: CPUBuffer):
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
    if (interface := getattr(buffer, "__cuda_array_interface__", None)) is not None:
        # Note: this is to silence pyre errors.
        # Usually, it should be asserting that `buffer` is a CUDABuffer,
        # but CUDABuffer class is a stub, so it would fail.
        assert not isinstance(buffer, CPUBuffer)
        if any(s == 0 for s in interface.get("shape", [])):
            raise ValueError("0-element array is not supported.")

        data_ptr = interface["data"][0]
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


def to_jax(buffer: CPUBuffer):
    """Convert to JAX Array.

    Args:
        buffer: Object implements the array interface protocol.

    Returns:
        (jax.Array): A JAX Array.
    """
    return jax.numpy.array(buffer, copy=False)
