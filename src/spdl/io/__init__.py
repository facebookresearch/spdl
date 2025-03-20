# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Implements the core I/O functionalities."""

# pyre-unsafe

from typing import Any

# This has to happen before other sub modules are imporeted.
# Otherwise circular import would occur.
from ._type_stub import (  # isort: skip
    CPUBuffer,
    CUDABuffer,
    Packets,
    AudioCodec,
    VideoCodec,
    ImageCodec,
    AudioPackets,
    VideoPackets,
    ImagePackets,
    Frames,
    AudioFrames,
    VideoFrames,
    ImageFrames,
    DemuxConfig,
    DecodeConfig,
    EncodeConfig,
    CUDAConfig,
    CPUStorage,
)

from ._array import (
    load_npy,
)
from ._composite import (
    load_audio,
    load_image,
    load_image_batch,
    load_image_batch_nvjpeg,
    load_video,
    sample_decode_video,
)
from ._config import (
    cpu_storage,
    cuda_config,
    decode_config,
    demux_config,
    encode_config,
)
from ._convert import (
    to_jax,
    to_numba,
    to_numpy,
    to_torch,
)
from ._core import (
    apply_bsf,
    convert_array,
    convert_frames,
    decode_image_nvjpeg,
    decode_packets,
    decode_packets_nvdec,
    demux_audio,
    demux_image,
    demux_video,
    Demuxer,
    encode_image,
    nv12_to_bgr,
    nv12_to_rgb,
    NvDecDecoder,
    streaming_decode_packets,
    transfer_buffer,
    transfer_buffer_cpu,
)
from ._preprocessing import (
    get_audio_filter_desc,
    get_filter_desc,
    get_video_filter_desc,
)
from ._zip import (
    load_npz,
    NpzFile,
)

__all__ = [
    # HIGH LEVEL API
    "load_audio",
    "load_video",
    "load_image",
    "load_image_batch",
    "load_image_batch_nvjpeg",
    "sample_decode_video",
    # Metadata
    "AudioCodec",
    "VideoCodec",
    "ImageCodec",
    # DEMUXING
    "Demuxer",
    "demux_audio",
    "demux_video",
    "demux_image",
    "Packets",
    "AudioPackets",
    "VideoPackets",
    "ImagePackets",
    "apply_bsf",
    # DECODING
    "decode_packets",
    "NvDecDecoder",
    "decode_packets_nvdec",
    "streaming_decode_packets",
    "decode_image_nvjpeg",
    "NvDecDecoder",
    "Frames",
    "AudioFrames",
    "VideoFrames",
    "ImageFrames",
    # PREPROCESSING
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
    # COLORSPACE CONVERSION
    "nv12_to_rgb",
    "nv12_to_bgr",
    # CAST
    "to_numba",
    "to_numpy",
    "to_torch",
    "to_jax",
    # ENCODING
    "encode_image",
    # CONFIG
    "demux_config",
    "DemuxConfig",
    "decode_config",
    "DecodeConfig",
    "encode_config",
    "EncodeConfig",
    "cuda_config",
    "CUDAConfig",
    "cpu_storage",
    "CPUStorage",
    # NUMPY
    "NpzFile",
    "load_npz",
    "load_npy",
]


def __dir__():
    return __all__


def __getattr__(name: str) -> Any:
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("spdl.io")
        except PackageNotFoundError:
            return "unknown"

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
