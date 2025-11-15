# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Implements the core I/O functionalities."""

# pyre-strict

from ._array import (
    load_npy,
    load_npz,
    NpzFile,
)
from ._composite import (
    load_audio,
    load_image,
    load_image_batch,
    load_image_batch_nvjpeg,
    load_video,
    sample_decode_video,
    save_image,
    streaming_load_video_nvdec,
)
from ._config import (
    audio_encode_config,
    cpu_storage,
    cuda_config,
    decode_config,
    demux_config,
    video_encode_config,
)
from ._convert import (
    ArrayInterface,
    CUDAArrayInterface,
    to_jax,
    to_numba,
    to_numpy,
    to_torch,
)
from ._core import (
    apply_bsf,
    BSF,
    convert_array,
    convert_frames,
    create_reference_audio_frame,
    create_reference_video_frame,
    decode_image_nvjpeg,
    decode_packets,
    decode_packets_nvdec,
    Decoder,
    demux_audio,
    demux_image,
    demux_video,
    Demuxer,
    encode_image,
    Muxer,
    nv12_to_bgr,
    nv12_to_rgb,
    nvdec_decoder,
    NvDecDecoder,
    transfer_buffer,
    transfer_buffer_cpu,
)
from ._preprocessing import (
    FilterGraph,
    get_abuffer_desc,
    get_audio_filter_desc,
    get_buffer_desc,
    get_filter_desc,
    get_video_filter_desc,
)
from ._tar import iter_tarfile
from ._transfer import transfer_tensor
from ._wav import (
    load_wav,
)

__all__ = [
    # HIGH LEVEL API
    "load_wav",
    "load_audio",
    "load_video",
    "load_image",
    "load_image_batch",
    "load_image_batch_nvjpeg",
    "sample_decode_video",
    "save_image",
    # Metadata
    "AudioCodec",
    "VideoCodec",
    "ImageCodec",
    # DEMUXING
    "Demuxer",
    "demux_audio",
    "demux_video",
    "demux_image",
    "AudioPackets",
    "VideoPackets",
    "ImagePackets",
    # BIT STREAM FILTERING
    "apply_bsf",
    "BSF",
    # DECODING
    "Decoder",
    "AudioDecoder",
    "VideoDecoder",
    "ImageDecoder",
    "decode_packets",
    "NvDecDecoder",
    "decode_packets_nvdec",
    "streaming_load_video_nvdec",
    "decode_image_nvjpeg",
    "NvDecDecoder",
    "nvdec_decoder",
    "AudioFrames",
    "VideoFrames",
    "ImageFrames",
    # FILTER GRAPH
    "FilterGraph",
    "get_abuffer_desc",
    "get_buffer_desc",
    "get_audio_filter_desc",
    "get_video_filter_desc",
    "get_filter_desc",
    # FRAME CONVERSION
    "convert_array",
    "convert_frames",
    "CPUBuffer",
    "CUDABuffer",
    # DATA TRANSFER
    "transfer_buffer",
    "transfer_buffer_cpu",
    "transfer_tensor",
    # COLORSPACE CONVERSION
    "nv12_to_rgb",
    "nv12_to_bgr",
    # CAST
    "ArrayInterface",
    "CUDAArrayInterface",
    "to_numba",
    "to_numpy",
    "to_torch",
    "to_jax",
    # ENCODING
    "Muxer",
    "AudioEncoder",
    "VideoEncoder",
    "create_reference_audio_frame",
    "create_reference_video_frame",
    "encode_image",
    # CONFIG
    "demux_config",
    "DemuxConfig",
    "decode_config",
    "DecodeConfig",
    "VideoEncodeConfig",
    "video_encode_config",
    "AudioEncodeConfig",
    "audio_encode_config",
    "cuda_config",
    "CUDAConfig",
    "cpu_storage",
    "CPUStorage",
    # NUMPY
    "NpzFile",
    "load_npz",
    "load_npy",
    # WAV AUDIO
    # Archive
    "iter_tarfile",
]


def __dir__() -> list[str]:
    return __all__


def __getattr__(name: str) -> object:
    """Lazily import C++ extension modules and their contents."""
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("spdl.io")
        except PackageNotFoundError:
            return "unknown"

    # Lazy loading of C++ extension classes from _libspdl
    _libspdl_items = {
        "CPUStorage",
        "CPUBuffer",
        "AudioCodec",
        "VideoCodec",
        "ImageCodec",
        "AudioPackets",
        "VideoPackets",
        "ImagePackets",
        "AudioFrames",
        "VideoFrames",
        "ImageFrames",
        "AudioEncoder",
        "VideoEncoder",
        "AudioDecoder",
        "VideoDecoder",
        "ImageDecoder",
        "DemuxConfig",
        "DecodeConfig",
        "VideoEncodeConfig",
        "AudioEncodeConfig",
    }

    if name in _libspdl_items:
        from . import lib

        disabled = lib._LG.disabled
        lib._LG.disabled = True
        try:
            attr = getattr(lib._libspdl, name)
            return attr
        except RuntimeError:

            class _placeholder:
                def __init__(self, *_args: object, **_kwargs: object) -> None:
                    raise RuntimeError(
                        f"Failed to load `_libspdl.{name}`. " "Is FFmpeg available?"
                    )

            return _placeholder

        finally:
            lib._LG.disabled = disabled

    # Lazy loading of C++ extension classes from _libspdl_cuda
    _libspdl_cuda_items = {
        "CUDABuffer",
        "CUDAConfig",
    }

    if name in _libspdl_cuda_items:
        from . import lib

        disabled = lib._LG.disabled
        lib._LG.disabled = True
        try:
            from .lib import _libspdl_cuda

            attr = getattr(_libspdl_cuda, name)
            return attr
        except RuntimeError:

            class _placeholder:
                def __init__(self, *_args: object, **_kwargs: object) -> None:
                    raise RuntimeError(
                        f"Failed to load `_libspdl_cuda.{name}`. "
                        "Is CUDA runtime available?"
                    )

            return _placeholder

        finally:
            lib._LG.disabled = disabled

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
