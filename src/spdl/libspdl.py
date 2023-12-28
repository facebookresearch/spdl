"""Thin wrapper around the libspdl extension."""

import sys
from typing import Any, List, Optional

import numpy as np
from numpy.typing import NDArray

from spdl.lib import libspdl as _libspdl


__all__ = [  # noqa: F822
    "init_folly",
    "Engine",
    "to_numpy",
]


def __dir__() -> List[str]:
    return sorted(__all__)


def __getattr__(name: str) -> Any:
    return getattr(_libspdl, name)


def init_folly(args: List[str]) -> List[str]:
    """Initialize folly internal mechanisms like singletons and logging."""
    return _libspdl.init_folly(sys.argv[0], args)[1:]


def to_numpy(buffer, format: Optional[str] = "NCHW") -> NDArray:
    """Convert to numpy array.

    Args:
        buffer (VideoBuffer): Raw buffer.

        format (str or None): Channel order.
            Valid values are ``"NCHW"``, ``"NHWC"`` or ``None``.
            If ``None`` no conversion is performed and native format is returned.
    """
    if buffer.is_cuda():
        raise RuntimeError("CUDA frames cannot be converted to numpy array.")
    array = np.array(buffer, copy=False)
    match format:
        case "NCHW":
            if buffer.channel_last:
                array = np.moveaxis(array, -1, -3)
        case "NHWC":
            if not buffer.channel_last:
                array = np.moveaxis(array, -3, -1)
        case None:
            pass
        case _:
            raise ValueError(
                "Expected format value to be one of ['NCHW', 'NHWC', None]"
                f", but received: {format}"
            )
    return array


class Frames:
    def __init__(self, frames):
        self._frames = frames

    def __getattr__(self, name):
        return getattr(self._frames, name)

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, slice):
        return self._frames[slice]

    def to_video_buffer(self, plane=-1):
        buffer = self._frames.to_video_buffer(plane)
        return VideoBuffer(buffer)


class VideoBuffer:
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


class Engine:
    def __init__(self, *args, **kwargs):
        self._engine = _libspdl.Engine(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    def dequeue(self):
        frames = self._engine.dequeue()
        return Frames(frames)
