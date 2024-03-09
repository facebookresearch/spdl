"""Thin wrapper around the libspdl extension."""

import sys
from typing import Any, List

from spdl.lib import libspdl as _libspdl

__all__ = [  # noqa: F822
    "BasicAdoptor",
    "Buffer",
    "BytesAdoptor",
    "CPUBuffer",
    "CUDABuffer",
    "CUDABuffer2DPitch",
    "DecodedFrames",
    "FFmpegAudioFrames",
    "FFmpegImageFrames",
    "FFmpegVideoFrames",
    "MMapAdoptor",
    "MultipleDecodingResult",
    "NvDecVideoFrames",
    "SingleDecodingResult",
    "SourceAdoptor",
    "TracingSession",
    "batch_decode_image",
    "batch_decode_image_nvdec",
    "clear_ffmpeg_cuda_context_cache",
    "convert_to_buffer",
    "convert_to_cpu_buffer",
    "create_cuda_context",
    "decode_audio",
    "decode_image",
    "decode_image_nvdec",
    "decode_video",
    "decode_video_nvdec",
    "get_ffmpeg_log_level",
    "init_folly",
    "init_tracing",
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
