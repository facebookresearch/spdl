"""Thin wrapper around the libspdl extension."""

import sys
from typing import Any, List, Optional

import numpy as np

try:
    from numpy.typing import NDArray
except ImportError:
    from numpy import ndarray as NDArray

from spdl.lib import libspdl as _libspdl


__all__ = [  # noqa: F822
    "BasicAdoptor",
    "Buffer",
    "DecodedFrames",
    "MMapAdoptor",
    "SourceAdoptor",
    "clear_ffmpeg_cuda_context_cache",
    "convert_frames",
    "create_cuda_context",
    "decode_audio",
    "decode_video",
    "get_ffmpeg_log_level",
    "init_folly",
    "set_ffmpeg_log_level",
]


def __dir__() -> List[str]:
    return sorted(__all__)


def __getattr__(name: str) -> Any:
    if name == "SourceAdoptor":
        return _libspdl.SourceAdoptor_SPDL_GLOBAL
    return getattr(_libspdl, name)


def init_folly(args: List[str]) -> List[str]:
    """Initialize folly internal mechanisms like singletons and logging."""
    return _libspdl.init_folly(sys.argv[0], args)[1:]


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
    return _BufferWrapper(_libspdl.convert_to_cpu_buffer(frames, index))


def to_numpy(frames, index: Optional[int] = None) -> NDArray:
    """Convert to numpy array.

    Args:
        frames (DecodedFrames): Decoded frames.

        index (int or None): The index of plane to be included in the output.
            For formats like YUV420, in which chroma planes have different sizes
            than luma plane, this argument can be used to select a specific plane.
    """
    return np.array(_to_cpu_buffer(frames, index), copy=False)
