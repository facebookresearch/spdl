"""Thin wrapper around the libspdl extension."""

import sys
from typing import Any, List

from spdl.lib import libspdl as _libspdl

__all__ = [  # noqa: F822
    "AudioPackets",
    "Buffer",
    "BytesAdaptor",
    "CPUBuffer",
    "CUDABuffer",
    "CUDABuffer2DPitch",
    "FFmpegAudioFrames",
    "FFmpegImageFrames",
    "FFmpegVideoFrames",
    "Future",
    "ImagePackets",
    "InternalError",
    "MMapAdaptor",
    "NvDecImageFrames",
    "NvDecVideoFrames",
    "SourceAdaptor",
    "ThreadPoolExecutor",
    "TracingSession",
    "VideoPackets",
    "async_apply_bsf",
    "async_convert_audio",
    "async_convert_audio_cpu",
    "async_convert_batch_image",
    "async_convert_batch_image_nvdec",
    "async_convert_image",
    "async_convert_image_cpu",
    "async_convert_image_nvdec",
    "async_convert_video",
    "async_convert_video_cpu",
    "async_convert_video_nvdec",
    "async_decode_audio",
    "async_decode_image",
    "async_decode_image_nvdec",
    "async_decode_video",
    "async_decode_video_nvdec",
    "async_demux_audio",
    "async_demux_audio_bytes",
    "async_demux_image",
    "async_demux_image_bytes",
    "async_demux_video",
    "async_demux_video_bytes",
    "async_sleep",
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
    "get_cuda_device_index",
    "get_ffmpeg_log_level",
    "init_folly",
    "init_tracing",
    "set_ffmpeg_log_level",
    "trace_counter",
    "trace_default_decode_executor_queue_size",
    "trace_default_demux_executor_queue_size",
    "trace_event_begin",
    "trace_event_end",
]


def __dir__() -> List[str]:
    return sorted(__all__)


def __getattr__(name: str) -> Any:
    return getattr(_libspdl, name)


def init_folly(args: List[str]) -> List[str]:
    """Initialize folly internal mechanisms like singletons and logging."""
    return _libspdl.init_folly(sys.argv[0], args)[1:]
