"""Thin wrapper around the libspdl extension."""

import sys
from typing import Any, List, Optional

import numpy as np
from numpy.typing import NDArray

from spdl.lib import libspdl as _libspdl


__all__ = [  # noqa: F822
    "decode_video",
    "init_folly",
    "to_numpy",
]


def __dir__() -> List[str]:
    return sorted(__all__)


def __getattr__(name: str) -> Any:
    return getattr(_libspdl, name)


def init_folly(args: List[str]) -> List[str]:
    """Initialize folly internal mechanisms like singletons and logging."""
    return _libspdl.init_folly(sys.argv[0], args)[1:]


class _BufferWrapper:
    def __init__(self, buffer):
        self._buffer = buffer

    def __getattr__(self, name):
        if name == "__array_interface__":
            if not self._buffer.is_cuda():
                return self._buffer.get_array_interface()
        if name == "__cuda_array_interface__":
            if self._buffer.is_cuda():
                return self._buffer.get_cuda_array_interface()
        return getattr(self._buffer, name)


def to_numpy(
    frames, format: Optional[str] = None, index: Optional[int] = None
) -> NDArray:
    """Convert to numpy array.

    Args:
        frames (FrameContainer): Decoded frames.

        format (str or None): Channel order.
            Valid values are ``"channel_first"``, ``"channel_last"`` or ``None``.
            If ``None`` no conversion is performed and native format is returned.
            If ``"channel_first"``, the returned video data is "NCHW".
            If ``"channel_last"``, the returned video data is "NHWC".

            (``"NCHW"`` and ``"NHWC"`` can be  respectively used alias for
             ``"channel_first"`` and ``"channel_last"`` in case of video frames.)
    """
    if frames.is_cuda():
        raise RuntimeError("CUDA frames cannot be converted to numpy array.")

    buffer = _BufferWrapper(frames.to_buffer(index))
    array = np.array(buffer, copy=False)
    match format:
        case "channel_first" | "NCHW":
            if buffer.channel_last:
                if frames.media_type == "video":
                    array = np.moveaxis(array, -1, -3)
                else:
                    array = np.moveaxis(array, -1, -2)
        case "channel_last" | "NHWC":
            if not buffer.channel_last:
                if frames.media_type == "video":
                    array = np.moveaxis(array, -3, -1)
                else:
                    array = np.moveaxis(array, -2, -1)
        case None:
            pass
        case _:
            raise ValueError(
                'Expected `format` argument to be one of ``"channel_first"``, '
                f'``"channel_last"``, ``None``. Found: {format}'
            )
    return array
