"""Thin wrapper around the libspdl extension."""

import sys
from typing import Any, List

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
