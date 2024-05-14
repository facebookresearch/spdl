from typing import TypeVar

from spdl.lib import _libspdl

from ._async import _async_task

__all__ = [
    "async_encode_image",
]

Array = TypeVar("Array")

async def async_encode_image(dst: str, data: Array, pix_fmt: str="rgb24", **kwargs):
    """Save the given image array/tensor to file.

    Args:
        dst: The path to which the data are written.

        data (NumPy NDArray, PyTorch Tensor):
            Image data in array format. The data  must be `uint8` type,
            either on CPU or CUDA device.

            The shape must be one of the following and must match the
            value of `pix_fmt`.

            - `(height, width, channel==3)` when `pix_fmt="rgb24"`
            - `(height, width)` when `pix_fmt=gray8`
            - `(channel==3, height, width)` when `pix_fmt="yuv444p"`

        pix_fmt: See above.

    Other args:
        encode_config (EncodeConfig): Customize the encoding.

        executor (Executor): Thread pool executor to run the operation.
            By default, it uses the default encode thread pool.

    ??? note "Example - Save image as PNG with resizing"

        ```python
        >>> import asyncio
        >>> import numpy as np
        >>> import spdl.io
        >>>
        >>> data = np.random.randint(255, size=(32, 16, 3), dtype=np.uint8)
        >>> coro = spdl.io.async_encode_image(
        ...     "foo.png",
        ...     data,
        ...     pix_fmt="rgb24",
        ...     encode_config=spdl.io.encode_config(width=198, height=96, scale_algo="neighbor"),
        ... )
        >>> asyncio.run(coro)
        >>>
        ```
    
    ??? note "Example - Directly save CUDA tensor as image"

        ```python
        >>> import torch
        >>>
        >>> data = torch.randint(255, size=(32, 16, 3), dtype=torch.uint8, device="cuda")
        >>> coro = spdl.io.async_encode_image(
        ...     "foo.png",
        ...     data,
        ...     pix_fmt="rgb24",
        ... )
        >>> asyncio.run(coro)
        >>>
        ```
    
    """
    func = _libspdl.async_encode_image
    await _async_task(func, dst, data, pix_fmt=pix_fmt, **kwargs)
