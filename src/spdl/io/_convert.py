# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Protocol, runtime_checkable

from spdl.io import CUDABuffer

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
    "ArrayInterface",
    "CUDAArrayInterface",
]


@runtime_checkable
class ArrayInterface(Protocol):
    """ArrayInterface()

    Protocol for objects that implement the NumPy Array Interface Protocol.

    Objects implementing this protocol expose their data through the
    ``__array_interface__`` attribute, allowing zero-copy conversion to NumPy arrays.

    See: https://numpy.org/doc/stable/reference/arrays.interface.html
    """

    @property
    def __array_interface__(self) -> dict[str, Any]:
        """Return a dictionary containing array interface metadata.

        The dictionary must contain:
            - ``version``: Protocol version (3)
            - ``shape``: Tuple of array dimensions
            - ``typestr``: Data type string
            - ``data``: Tuple of (data_pointer, read_only_flag)
            - ``owner``: Optional object owning the data buffer

        Returns:
            Dictionary with array interface metadata.
        """
        ...


@runtime_checkable
class CUDAArrayInterface(Protocol):
    """CUDAArrayInterface()

    Protocol for objects that implement the CUDA Array Interface Protocol.

    Objects implementing this protocol expose their CUDA data through the
    ``__cuda_array_interface__`` attribute, allowing zero-copy conversion to
    CUDA-aware libraries like PyTorch and Numba.

    See: https://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html
    """

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        """Return a dictionary containing CUDA array interface metadata.

        The dictionary must contain:
            - ``version``: Protocol version (typically 3)
            - ``shape``: Tuple of array dimensions
            - ``typestr``: Data type string
            - ``data``: Tuple of (data_pointer, read_only_flag)
            - ``strides``: Optional tuple of strides (in bytes)
            - ``descr``: Optional data type descriptor
            - ``mask``: Optional mask array interface

        Returns:
            Dictionary with CUDA array interface metadata.
        """
        ...


def to_numpy(buffer: ArrayInterface):
    """Convert to NumPy NDArray.

    Args:
        buffer: Object that implements the array interface protocol
        (has ``__array_interface__``).

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


def to_torch(buffer: ArrayInterface | CUDABuffer):
    """Convert to PyTorch Tensor.

    Args:
        buffer: Object that implements the array interface protocol
        (has ``__array_interface__``) or :py:class:`CUDABuffer` object.

    Returns:
        (torch.Tensor): A PyTorch Tensor.
    """
    if (interface := getattr(buffer, "__cuda_array_interface__", None)) is not None:
        # Note: this is to silence pyre errors.
        # Usually, it should be asserting that `buffer` is a CUDABuffer,
        # but CUDABuffer class is a stub, so it would fail.
        assert not isinstance(buffer, ArrayInterface)
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


def to_numba(buffer: ArrayInterface | CUDAArrayInterface):
    """Convert to Numba DeviceNDArray or NumPy NDArray.

    Args:
        buffer: Object that implements the array interface protocol
        (has ``__array_interface__``) or
        CUDA array interface protocol (has ``__cuda_array_interface__``).

    Returns:
        (DeviceNDArray or NDArray): A Numba DeviceNDArray or NumPy NDArray.

    See Also:
        https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
    """
    if hasattr(buffer, "__cuda_array_interface__"):
        return cuda.as_cuda_array(buffer)

    return np.array(buffer, copy=False)


def to_jax(buffer: ArrayInterface):
    """Convert to JAX Array.

    Args:
        buffer: Object that implements the array interface protocol
        (has ``__array_interface__``).

    Returns:
        (jax.Array): A JAX Array.
    """
    return jax.numpy.array(buffer, copy=False)
